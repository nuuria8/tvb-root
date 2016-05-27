# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)

"""
Models based on Wong-Wang's work.

"""

from .base import Model, LOG, numpy, basic, arrays


class ReducedWongWang(Model):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2013] Deco Gustavo, Ponce Alvarez Adrian, Dante Mantini, Gian Luca
                  Romani, Patric Hagmann and Maurizio Corbetta. *Resting-State
                  Functional Connectivity Emerges from Structurally and
                  Dynamically Shaped Slow Linear Fluctuations*. The Journal of
                  Neuroscience 32(27), 11239-11252, 2013.



    .. automethod:: ReducedWongWang.__init__

    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

    """
    _ui_name = "Reduced Wong-Wang"
    ui_configurable_parameters = ['a', 'b', 'd', 'gamma', 'tau_s', 'w', 'J_N', 'I_o']

    #Define traited attributes for this model, these represent possible kwargs.
    a = arrays.FloatArray(
        label=":math:`a`",
        default=numpy.array([0.270, ]),
        range=basic.Range(lo=0.0, hi=0.270, step=0.01),
        doc=""" [pC]^{-1}. Parameter chosen to fit numerical solutions.""",
        order=1)

    b = arrays.FloatArray(
        label=":math:`b`",
        default=numpy.array([0.108, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[kHz]. Parameter chosen to fit numerical solutions.""",
        order=2)

    d = arrays.FloatArray(
        label=":math:`d`",
        default=numpy.array([154., ]),
        range=basic.Range(lo=0.0, hi=200.0, step=0.01),
        doc="""[ms]. Parameter chosen to fit numerical solutions.""",
        order=3)

    gamma = arrays.FloatArray(
        label=r":math:`\gamma`",
        default=numpy.array([0.641, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Kinetic parameter""",
        order=4)

    tau_s = arrays.FloatArray(
        label=r":math:`\tau_S`",
        default=numpy.array([100., ]),
        range=basic.Range(lo=50.0, hi=150.0, step=1.0),
        doc="""Kinetic parameter. NMDA decay time constant.""",
        order=5)

    w = arrays.FloatArray(
        label=r":math:`w`",
        default=numpy.array([0.6, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Excitatory recurrence""",
        order=6)

    J_N = arrays.FloatArray(
        label=r":math:`J_{N}`",
        default=numpy.array([0.2609, ]),
        range=basic.Range(lo=0.2609, hi=0.5, step=0.001),
        doc="""Excitatory recurrence""",
        order=7)

    I_o = arrays.FloatArray(
        label=":math:`I_{o}`",
        default=numpy.array([0.33, ]),
        range=basic.Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input""",
        order=8)

    sigma_noise = arrays.FloatArray(
        label=r":math:`\sigma_{noise}`",
        default=numpy.array([0.000000001, ]),
        range=basic.Range(lo=0.0, hi=0.005),
        doc="""[nA] Noise amplitude. Take this value into account for stochatic
        integration schemes.""",
        order=-1)

    state_variable_range = basic.Dict(
        label="State variable ranges [lo, hi]",
        default={"S": numpy.array([0.0, 1.0])},
        doc="Population firing rate",
        order=9
    )

    variables_of_interest = basic.Enumerate(
        label="Variables watched by Monitors",
        options=["S"],
        default=["S"],
        select_multiple=True,
        doc="""default state variables to be monitored""",
        order=10)

    #    variables_of_interest = arrays.IntegerArray(
    #        label="Variables watched by Monitors",
    #        range=basic.Range(lo=0.0, hi=1.0, step=1.0),
    #        default=numpy.array([0], dtype=numpy.int32),
    #        doc="default state variables to be monitored",
    #        order=10)


    def __init__(self, **kwargs):
        """
        .. May need to put kwargs back if we can't get them from trait...

        """

        #LOG.info('%s: initing...' % str(self))

        super(ReducedWongWang, self).__init__(**kwargs)

        #self._state_variables = ["S1"]
        self._nvar = 1
        self.cvar = numpy.array([0], dtype=numpy.int32)

        LOG.debug('%s: inited.' % repr(self))

    def configure(self):
        """  """
        super(ReducedWongWang, self).configure()
        self.update_derived_parameters()


    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Equations taken from [DPA_2013]_ , page 11242

        .. math::
                 x_k       &=   w\,J_N \, S_k + I_o + J_N \mathbf\Gamma(S_k, S_j, u_{kj}),\\
                 H(x_k)    &=  \dfrac{ax_k - b}{1 - \exp(-d(ax_k -b))},\\
                 \dot{S}_k &= -\dfrac{S_k}{\tau_s} + (1 - S_k) \, H(x_k) \, \gamma

        """

        S   = state_variables[0, :]
        S[S<0] = 0.
        S[S>1] = 1.
        c_0 = coupling[0, :]


        # if applicable
        lc_0 = local_coupling * S

        x  = self.w * self.J_N * S + self.I_o + self.J_N * c_0 + self.J_N * lc_0
        H = (self.a * x - self.b) / (1 - numpy.exp(-self.d * (self.a * x - self.b)))
        dS = - (S / self.tau_s) + (1 - S) * H * self.gamma

        derivative = numpy.array([dS])
        return derivative

