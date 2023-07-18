import numpy as np
from njet.functions import cos
from lieops import construct

from .common import hard_edge_element
from .drift import DriftHamiltonian

class rfcavity(hard_edge_element):
    def __init__(self, voltage, phase, frequency, beta0, *args, **kwargs):
        '''
        A generic RF cavity.
                
        Reference(s):
        [1] A. Wolski: Beam Dynamics in High Energy Particle Accelerators.
        '''
        assert 0 < beta0 and beta0 < 1
        self.voltage = voltage
        self.phase = phase
        self.frequency = frequency # Actually, translated to: k = 2*np.pi/constants.speed_of_light*1/T
        self.beta0 = beta0
        hard_edge_element.__init__(self, *args, beta0=beta0, **kwargs)
        
    def calcHamiltonian(self, p=10, tol_drop=0, **kwargs):
        '''
        The Hamiltonian of a simplified RF cavity., see Ref. [1], Eq. (3.138) p. 112.
        
        Parameters
        ----------
        p: int, optional
            The total order up to which the RF potential should be expanded in terms of polynomials.
        
        tol_drop: float, optional
            Threshold below which terms in the Hamiltonian are considered to be zero.
        '''
        DH = DriftHamiltonian(tol_drop=tol_drop, **kwargs)
        x, y, sigma, px, py, psigma = DH['coords']
        #k = 2*np.pi*self.frequency/constants.speed_of_light # 2.40483/radius # Eq. (3.132) in [1] and earlier: omega = k/c
        #hamiltonian = construct(cos, sigma/beta0*-k + self.phase, **kwargs)*self.voltage
        rf_potential = construct(cos, -sigma/self.beta0*self.frequency + self.phase, power=p, **kwargs)*self.voltage/float(np.pi)
        hamiltonian = DH['full'] - rf_potential
        self.hamiltonian = hamiltonian.pop((0, 0, 0, 0, 0, 0), None).above(tol_drop) # remove any constant term

