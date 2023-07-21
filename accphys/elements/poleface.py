import numpy as np
from scipy.linalg import logm

from lieops.core.tools import ad2poly

from .common import hard_edge_element

class polefaceRM(hard_edge_element):

    def __init__(self, rho, phi=0, **kwargs):
        '''
        The model of a pole-face-rotation-matrix according to Ref. [1]. This element can be used to 
        transform an S-bend approximatively into a rectangular bend, see Ref. [1].
        
        Parameters
        ----------
        rho: float
            The bending radius of the underlying dipole.
            
        phi: float
            The angle of the pole-face relative to the transverse coordinates in the Frenet-Serret system.
        
        Reference(s)
        ------------
        [1] K. Brown: "Optics and Lattices" in: A. W. Chao and M. Tigner: "Handbook of Accelerator Physics
            and Engineering", 3rd edition, World Scientific 1998. 
        '''
        assert rho != 0
        
        self.rho = rho
        self.phi = phi
        self._htp = np.tan(phi)/self.rho
        self.length = 0
        hard_edge_element.__init__(self, **kwargs)
    
    def calcHamiltonian(self, **kwargs):
        
        pfrm = [[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [self._htp, 0, 0, 1, 0, 0],
                [0, -self._htp, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]]
                
        h0 = logm(np.array(pfrm).transpose()) # The reason why the transpose has been used is explained in lieops.core.dragt.py
        self.hamiltonian = ad2poly(h0)
        
        