from lieops import lexp

from .common import hard_edge_element
from .cfm import cfm
from .poleface import polefaceRM

class rcfmRM(hard_edge_element):
    '''
    An implementation of a 'rectangular' CFMs by using the
    rotation-matrix (RM) approximation in Brown (see accphys.elements.poleface.py)
    to model the sector bend wedges infront and after the cfm (relative to the horizontal
    plane). The overall Hamiltonian will then be calculated by using Hadamard's Theorem.

    We emphasize that in this way higher-order components in the CFM in these fringe
    sectors are *not* included.
    '''
    def __init__(self, *args, **kwargs):
        self.inner = cfm(*args, **kwargs)
        self.wedge = polefaceRM(rho=self.inner.rhox, phi=self.inner.phix/2)
        self.hamiltonian = lexp(-self.wedge.hamiltonian)(self.inner.hamiltonian, method='channell') # any length of the interior will be taken care of in the final 'setOperator' routine upon the initialization of the hard_edge_element class below:
        hard_edge_element.__init__(self, *args, **kwargs)
