from functools import reduce

from accphys.tools import _depsilon, detuning
from lieops import poly, lexp, create_coords

from .tools import f_identity, f_compose

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
        
        Parameters
        ----------
        **kwargs
            Optional arguments passed to self.hamiltonian.bnf routine.
        '''
        # power should be sufficiently high; self.lo_power = 10 is not sufficient for nfRot. Better 20.
        # TODO:
        # - may use self.lo_power instead.
        # - replace steps by symplectic integrator to prevent the use of max_power etc.
        # - Only the action on xi may be necessary, but currently I keep the full range to avoid any unecessary bugs.
        self._lo_power = 20
        
        self.__dict__.update(self.hamiltonian.bnf(**kwargs))
        self.detuning = detuning(self.__dict__)
        self.dHdaction = [poly(values=_depsilon(self.detuning, k), dim=self.dim) for k in range(self.dim)]
        # we use poly objects here, because in the next line we multiply these polynomials with other terms.
        # Also, we want to apply the resulting function on coordinates.
        
        _xieta = create_coords(dim=self.dim, max_power=self.hamiltonian.max_power)
        self._lHnfmap = lexp(poly(values=self.detuning, dim=self.dim), t=-.1, power=self._lo_power)(*_xieta)
        self.nfRot = lambda *z: [lhfm(*z) for lhfm in self._lHnfmap] # N.B. in general it is important that both xi and eta components are returned here.
        # alternative (requires exp with multiplication (not ad)):
        #self.nfRot = lambda *xieta: [exp(self.dHdaction[k]*1j, power=self.lo_power)*xieta[k] for k in range(self.dim)] +\
        #                            [exp(self.dHdaction[k]*-1j, power=self.lo_power)*xieta[k + self.dim] for k in range(self.dim)] 
        # (1.48) in Ref. [1] for the case of a single 'a' or 'b'-value (i.e. a_k = 1)
        
        self._lchi = [lexp(chi, t=1, power=self._lo_power) for chi in self.chi] # the Lie-operators belonging to the chi-transformations
        self._lchi_inv = [lexp(chi, t=-1, power=self._lo_power) for chi in self.chi] # the inverse Lie-operators belonging to the chi-transformations
        
        self._Amap = reduce(f_compose, self._lchi[::-1], f_identity)(*_xieta) # the map from ordinary (xi, eta)-space to normal-form space 
        self._A_invmap = reduce(f_compose, self._lchi_inv, f_identity)(*_xieta) # the map from normal-form space to ordinary (xi, eta)-space
        self.A = lambda *xieta: [A(*xieta) for A in self._Amap]  # N.B. in general it is important that both xi and eta components are returned here.
        self.A_inv = lambda *xieta: [Ai(*xieta) for Ai in self._A_invmap]  # N.B. in general it is important that both xi and eta components are returned here.

    def multiTurnMap(self, *xieta, n_reps=1):
        points = [xieta] # = [self.A_inv(xi_nf)] with xi_nf = self.A(*xieta)
        xi_nf = self.A(*xieta)
        for k in range(n_reps):
            xi_nf = self.nfRot(*xi_nf)
            points.append(self.A_inv(*xi_nf))
        return points
        
    def oneTurnMap(self, *xieta):
        return self.multiTurnMap(*xieta, n_reps=1)[-1]
    
    def __call__(self, *xieta):
        return self.oneTurnMap(*xieta)
    