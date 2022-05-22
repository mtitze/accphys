from functools import reduce
import numpy as np

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
        self.nf_setup(**kwargs)
        
    def nf_setup(self, t_nf=-1, **kwargs):
        '''
        Calculate the Birkhoff normal form of the given Hamiltonian, up to a specific order.
        
        Parameters
        ----------
        t_nf: float, optional
            The length of the integration step for the normal form Hamiltonian.
        
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
        self._lHnfmap = lexp(poly(values=self.detuning, dim=self.dim), t=t_nf, power=self._lo_power)(*_xieta)
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
        
        self.transform(**kwargs)

    def transform(self, kind='cnf', **kwargs):
        '''
        Select a linear transformation which should be executed when tracking before and after the non-linear maps to NF.
        
        Parameters
        ----------
        kind: str, optional
            A string denoting the nature of the transformation. Supported type(s) are:
            'cnf': Complex normal form
            'ops': Ordinary phase space
        '''
        supported_types = ['cnf', 'ops']
        if kind == 'cnf':
            # Complex normal form
            self.setTransform(np.eye(self.dim*2))
        elif kind == 'ops':
            # Ordinary phase space
            self.setTransform(self.nfdict['K'], self.nfdict['Kinv'])
        else:
            raise RuntimeError(f'{kind} not recognized. Supported type(s): {supported_types}')
        
    def setTransform(self, N, N_inv=[]):
        '''
        Set linear transformation maps which should be executed when tracking before and after the non-linear maps to NF.
        
        Parameters
        ----------
        N: array_like
            The linear map to be applied.
        
        N_inv: array_like, optional
            The linear map to be applied after the operations. If nothing specified, the inverse of N will be used. 
        '''
        assert N.shape == (self.dim*2, self.dim*2)
        if len(N_inv) == 0:
            N_inv = np.linalg.inv(N)
        self._transform_A2 = lambda *z: [sum([z[l]*N[k, l] for l in range(self.dim*2)]) for k in range(self.dim*2)] # we use matrix multiplication by hand, to avoid numpy complains with nested arrays in case z[l] are numpy arrays; N[k, l] stands on the right here, to avoid possible numpy issues with right-multiplications
        self._transform_A2_inv = lambda *z: [sum([z[l]*N_inv[k, l] for l in range(self.dim*2)]) for k in range(self.dim*2)]
        
    def multiTurnMap(self, *xieta, n_reps=1, **kwargs):
        self.transform(**kwargs)
        
        points = [self._transform_A2_inv(*self._transform_A2(*xieta))] # A2 and back again ensures the output has the right format (e.g. if one component is given in terms of a numpy array and the other component a float etc.
        xieta = self._transform_A2(*xieta) # apply the linear map first
        xi_nf = self.A(*xieta)
        for k in range(n_reps):
            xi_nf = self.nfRot(*xi_nf)
            points.append(self._transform_A2_inv(*self.A_inv(*xi_nf)))
        return points
        
    def oneTurnMap(self, *xieta, **kwargs):
        kwargs['n_reps'] = 1
        return self.multiTurnMap(*xieta, **kwargs)[-1]
    
    def __call__(self, *xieta, **kwargs):
        return self.oneTurnMap(*xieta, **kwargs)
    