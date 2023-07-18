import numpy as np
import warnings
from scipy.linalg import logm

from njet.functions import cos
from lieops import create_coords, construct, poly, lexp
from lieops.solver.splitting import get_scheme_ordering
from lieops.solver import heyoka

from accphys.tools import energy2beta0

# Reference(s):
# [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
# [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)
        
class pfrm(hard_edge_element):

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
        assert rho > 0
        
        self.rho = rho
        self.phi = phi
        self._htp = np.tan(phi)/self.rho
        
        self.length = kwargs.get('length', 1) # for technical reasons (TODO: overwork parameters)
        
        self.calcHamiltonian()
    
    def calcHamiltonian(self):
        
        pfrm = [[1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [self._htp, 0, 0, 1, 0, 0],
                [0, -self._htp, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]]
                
        h0 = logm(np.array(pfrm).transpose()) # The reason why the transpose is required is explained in lieops.core.dragt.py
        self.full_hamiltonian = ad2poly(h0)
        
        
        
        
    

        
