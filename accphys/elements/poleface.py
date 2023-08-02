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
        self._tan_rho = np.tan(phi)/self.rho
        self.length = 0
        hard_edge_element.__init__(self, **kwargs)
    
    def calcHamiltonian(self, **kwargs):
        # Matrix according to Ref. [1], given in terms of (q, p)-coordinates
        self._matrix_qp = np.array([[1, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [self._tan_rho, 0, 0, 1, 0, 0],
                                    [0, -self._tan_rho, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
        
        # Transform the above matrix, into (xi, eta)-coordinates by means of the unitary matrix U:
        U = np.array([[1, 0, 0, 1j, 0, 0], 
                      [0, 1, 0, 0, 1j, 0],
                      [0, 0, 1, 0, 0, 1j],
                      [1, 0, 0, -1j, 0, 0],
                      [0, 1, 0, 0, -1j, 0],
                      [0, 0, 1, 0, 0, -1j]])/np.sqrt(2)
        self._matrix_xieta = U@self._matrix_qp@U.transpose().conjugate()
                
        h0 = logm(self._matrix_xieta.transpose()) # The reason why the transpose has been used is because the ad2poly routine (below) is given with respect to the transpose matrix (in order that the adjoint composition is compatible with matrix multiplication. See also the discussion and references in lieops.core.dragt.py).
        self.hamiltonian = -ad2poly(h0) 
        # Some remarks:
        # H0 := self.hamiltonian is the Hamiltonian in terms of complex xi/eta coordinates.
        # The equations of motion thus have the form
        # dz_j/dt = {z_j, H0}_(xi/eta)
        # Note that the 'real' Hamiltonian Hr would be related to H0 by Hr*-1j = H0, since the complex structure in 
        # xi/eta coordinates is differently:
        # dz_j/dt = -1j*{z_j, Hr}_(xi/eta)
        # For this reason, the minus sign in the above line is required, to transform into dz_j/dt = {-H0, z_j}.
        