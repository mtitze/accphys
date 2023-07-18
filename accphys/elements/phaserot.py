from lieops import create_coords

from .common import hard_edge_element

class phaserot(hard_edge_element):
    def __init__(self, *tunes, **kwargs):
        '''
        A generic uncoupled phase (space) rotation.
        
        Parameters
        ----------
        tunes: float or array_like
            Tune(s) defining the phase rotation.
        '''
        self.tunes = tunes
        if len(tunes) > 0:
            self.calcHamiltonian(*tunes, **kwargs)
        
    def calcHamiltonian(self, *tunes, **kwargs):
        dim = len(tunes)
        xieta = create_coords(dim=dim, **kwargs)
        self.hamiltonian = sum([-tunes[k]*xieta[k]*xieta[k + dim] for k in range(dim)])