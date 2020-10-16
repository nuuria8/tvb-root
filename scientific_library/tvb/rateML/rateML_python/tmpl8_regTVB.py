from tvb.simulator.models.base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class ${dfunname}(ModelNumbaDfun):
    %for mconst in const:
        ${NArray(mconst)}
    %endfor

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={\
%for itemA in dynamics.state_variables:
"${itemA.name}": numpy.array([${itemA.dimension}])${'' if loop.last else ', \n\t\t\t\t '}\
%endfor
},
        doc="""state variables"""
    )

% if svboundaries:
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={\
%for limit in dynamics.state_variables:
% if (limit.exposure!='None' and limit.exposure!=''):
"${limit.name}": numpy.array([${limit.exposure}])\
% endif
%endfor
},
    )
% endif \

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=(\
%for itemJ in exposures:
%if {loop.first}:
%for choice in (itemJ.dimension):
'${choice}', \
%endfor
),
        default=(\
%for defa in (itemJ.description):
'${defa}', \
%endfor
%endif
%endfor
),
        doc="${itemJ.description}"
    )

    state_variables = [\
%for itemB in dynamics.state_variables:
'${itemB.name}'${'' if loop.last else ', '}\
%endfor
]

    _nvar = ${dynamics.state_variables.__len__()}
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        ##lc_0 = local_coupling * vw[0, :, 0]
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_${dfunname}(vw_, c_, \
%for itemE in const:
self.${itemE.name}, \
%endfor
local_coupling)

        return deriv.T[..., numpy.newaxis]

## signature is always the number of constants +4. the extras are vw, c_0, lc_0 and dx.
@guvectorize([(float64[:], float64[:], \
% for i in range(const.__len__()+1):
float64, \
% endfor
float64[:])], '(n),(m)' + ',()'*${const.__len__()+1} + '->(n)', nopython=True)
def _numba_dfun_${dfunname}(vw, coupling, \
% for itemI in const:
${itemI.name}, \
% endfor
local_coupling, dx):
    "Gufunc for ${dfunname} model equations."

    % for i, itemF in enumerate(dynamics.state_variables):
    ${itemF.name} = vw[${i}]
    % endfor

% if (dynamics.derived_variables):
    # derived variables
    % for der_var in dynamics.derived_variables:
    ${der_var.name} = ${der_var.value}
    % endfor
%endif /

% if dynamics.conditional_derived_variables:
    # Conditional variables
    % for con_der in dynamics.conditional_derived_variables:
        % for case in (con_der.cases):
            % if (loop.first):
    if ${case.condition}:
        ${con_der.name} = ${case.value}
            % elif (not loop.last and not loop.first):
    elif ${case.condition}:
        ${con_der.name} = ${case.value}
            % elif (loop.last):
    else:
        ${con_der.name} = ${case.value}

            %endif
        % endfor
    % endfor
% endif /


    % for j, itemH in enumerate(dynamics.time_derivatives):
    dx[${j}] = ${itemH.value}
    % endfor
    \
    \
    ## TVB numpy constant declarations
    <%def name="NArray(nconst)">
    ${nconst.name} = NArray(
        label=":math:`${nconst.name}`",
        default=numpy.array([${nconst.value}]),
        % if (nconst.dimension != "None" and nconst.dimension != ""):
        domain=Range(${nconst.dimension}),
        % endif
        doc="""${nconst.description}"""
    )\
    ##self.${nconst.name} = ${nconst.name}
    </%def>