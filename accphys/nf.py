from njet.functions import exp
from functools import reduce

from accphys.tools import _depsilon, detuning
from lieops import poly, lexp, create_coords

# routines which will work with composition of Lie-operators applied on polynomials.
def lie_identity(x):
    return x

def lie_compose(f, g):
    return lambda x: f(g(x))

class nf:
    '''
    Class to handle operations related to the normal form of a given Hamiltonian.
    
    Reference(s):
    [1]: M. Titze, "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019).
    '''
    def __init__(self, hamiltonian, **kwargs):
        self.hamiltonian = hamiltonian
        self.dim = hamiltonian.dim
        self.set_order(**kwargs)
        
    def set_order(self, **kwargs):
        '''
        Calculate the Birkhoff normal form of the given Hamiltonian, up to a specific order.
        '''
        self.__dict__.update(self.hamiltonian.bnf(**kwargs))
        self.detuning = detuning(self.__dict__)
        self.dHdaction = [poly(values=_depsilon(self.detuning, k), dim=self.dim) for k in range(self.dim)]
        # we use poly objects here, because in the next line we multiply these polynomials with other terms.
        # Also, we want to apply the resulting function on coordinates.
        self.nfRot = lambda *xieta: [exp(self.dHdaction[k](*xieta)*1j)*xieta[k] for k in range(self.dim)] # (1.48) in Ref. [1] for the case of a single 'a'-value (=k)
        # Note that we decided to use a 'general' njet.functions.exp in order to use jets later on.
        
        self._lchi = [lexp(chi, t=1, power=self.lo_power) for chi in self.chi] # the Lie-operators belonging to the chi-transformations
        self._lchi_inv = [lexp(chi, t=-1, power=self.lo_power) for chi in self.chi] # the inverse Lie-operators belonging to the chi-transformations
        
        xi = create_coords(dim=self.dim)[:self.dim]
        # TODO: replace this by symplectic integrator to prevent the use of max_power etc.
        self.A = lambda *xieta: [(reduce(lie_compose, self._lchi[::-1], lie_identity)(x))(*xieta) for x in xi] # the map from ordinary (xi, eta)-space to normal-form space 
        self.A_inv = lambda *xieta: [(reduce(lie_compose, self._lchi_inv, lie_identity)(x))(*xieta) for x in xi] # the map from normal-form space to ordinary (xi, eta)-space

    def oneTurnMap(self, *xieta):
        xi_nf = self.A(*xieta)
        xi_nf_rotated = self.nfRot(*xi_nf)
        return self.A_inv(*xi_nf_rotated)

    def __call__(self, *xieta):
        return self.oneTurnMap(*xieta)
    