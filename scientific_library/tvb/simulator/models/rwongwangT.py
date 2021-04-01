from tvb.simulator.models.base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class RwongwangT(ModelNumbaDfun):
        
    w_plus = NArray(
        label=":math:`w_plus`",
        default=numpy.array([1.4]),
        doc=""""""
    )    
        
    a_E = NArray(
        label=":math:`a_E`",
        default=numpy.array([310.0]),
        doc=""""""
    )    
        
    b_E = NArray(
        label=":math:`b_E`",
        default=numpy.array([125.0]),
        doc=""""""
    )    
        
    d_E = NArray(
        label=":math:`d_E`",
        default=numpy.array([0.154]),
        doc=""""""
    )    
        
    a_I = NArray(
        label=":math:`a_I`",
        default=numpy.array([615.0]),
        doc=""""""
    )    
        
    b_I = NArray(
        label=":math:`b_I`",
        default=numpy.array([177.0]),
        doc=""""""
    )    
        
    d_I = NArray(
        label=":math:`d_I`",
        default=numpy.array([0.087]),
        doc=""""""
    )    
        
    gamma_E = NArray(
        label=":math:`gamma_E`",
        default=numpy.array([0.641 / 1000.0]),
        doc=""""""
    )    
        
    tau_E = NArray(
        label=":math:`tau_E`",
        default=numpy.array([100.0]),
        doc=""""""
    )    
        
    tau_I = NArray(
        label=":math:`tau_I`",
        default=numpy.array([10.0]),
        doc=""""""
    )    
        
    I_0 = NArray(
        label=":math:`I_0`",
        default=numpy.array([0.382]),
        doc=""""""
    )    
        
    w_E = NArray(
        label=":math:`w_E`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    w_I = NArray(
        label=":math:`w_I`",
        default=numpy.array([0.7]),
        doc=""""""
    )    
        
    gamma_I = NArray(
        label=":math:`gamma_I`",
        default=numpy.array([1.0 / 1000.0]),
        doc=""""""
    )    
        
    J_N = NArray(
        label=":math:`J_N`",
        default=numpy.array([0.15]),
        doc=""""""
    )    
        
    J_I = NArray(
        label=":math:`J_I`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0]),
        doc=""""""
    )    
        
    lamda = NArray(
        label=":math:`lamda`",
        default=numpy.array([0.0]),
        doc=""""""
    )    
        
    J_NMDA = NArray(
        label=":math:`J_NMDA`",
        default=numpy.array([0.15]),
        doc=""""""
    )    
        
    JI = NArray(
        label=":math:`JI`",
        default=numpy.array([1.0]),
        doc=""""""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([0.0, 1.0]), 
				 "W": numpy.array([0.0, 1.0])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"V": numpy.array([0.0, 1.0]), "W": numpy.array([0.0, 1.0]), },
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('V', 'W', ),
        default=('W', 'W', ),
        doc="Variables to monitor"
    )

    state_variables = ['V', 'W']

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_RwongwangT(vw_, c_, self.w_plus, self.a_E, self.b_E, self.d_E, self.a_I, self.b_I, self.d_I, self.gamma_E, self.tau_E, self.tau_I, self.I_0, self.w_E, self.w_I, self.gamma_I, self.J_N, self.J_I, self.G, self.lamda, self.J_NMDA, self.JI, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*21 + '->(n)', nopython=True)
def _numba_dfun_RwongwangT(vw, coupling, w_plus, a_E, b_E, d_E, a_I, b_I, d_I, gamma_E, tau_E, tau_I, I_0, w_E, w_I, gamma_I, J_N, J_I, G, lamda, J_NMDA, JI, local_coupling, dx):
    "Gufunc for RwongwangT model equations."

    # long-range coupling
    c_pop1 = coupling[0]
    c_pop2 = coupling[1]
    c_pop3 = coupling[2]
    c_pop4 = coupling[3]

    V = vw[0]
    W = vw[1]

    # derived variables
    min_d_E = -1.0 * d_E
    min_d_I = -1.0 * d_I
    imintau_E = -1.0 / tau_E
    imintau_I = -1.0 / tau_I
    w_E__I_0 = w_E * I_0
    w_I__I_0 = w_I * I_0
    G_J_NMDA = G*J_NMDA
    w_plus__J_NMDA = w_plus * J_NMDA
    tmp_I_E = a_E * (w_E__I_0 + w_plus__J_NMDA * V + c_pop1 - JI*W) - b_E
    tmp_H_E = tmp_I_E/(1.0-exp(min_d_E * tmp_I_E))
    tmp_I_I = (a_I*((w_I__I_0+(J_NMDA * V))-W))-b_I
    tmp_H_I = tmp_I_I/(1.0-exp(min_d_I*tmp_I_I))

    dx[0] = (imintau_E* V)+(tmp_H_E*(1-V)*gamma_E)
    dx[1] = (imintau_I* W)+(tmp_H_I*gamma_I)
    