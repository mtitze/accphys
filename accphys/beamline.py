from functools import reduce
from tqdm.auto import tqdm

from lieops import create_coords, combine

def f_identity(x):
    return x

def f_compose(f, g):
    return lambda x: f(g(x))

class beamline:
    
    def __init__(self, *sequence, **kwargs):
        '''
        Class to model an accelerator beamline.
        '''
        # consistency checks
        assert len(sequence) > 0
        assert all([hasattr(e, 'hamiltonian') for e in sequence])
        dim0 = sequence[0].hamiltonian.dim
        assert all([e.hamiltonian.dim == dim0 for e in sequence])
        
        self.dim = dim0
        self.sequence = sequence
        self.lengths = [e.length for e in sequence]
            
    def __len__(self):
        return len(self.sequence)
    
    def __getitem__(self, key):
        return self.sequence[key]
    
    def __setitem__(self, key, value):
        self.sequence[key] = value
        
    def calcFlows(self, t=-1, **kwargs):
        '''
        Calculate the sequence of Lie-operators exp(:f_n:) for each element f_n in self.sequence.
        
        Parameters
        ----------
        Parameters
        ----------
        lengths: list, optional
            Lengths of the elements. These lengths will modifying the respecive flows accordingly.
            If given, then exp(lengths[k]:x[k]:) will be computed. An additional t-argument will
            compute exp(t*lengths[k]:x[k]:).

        **kwargs
            Optional parameters passed to lieops.ops.lie.poly.flow
        '''
        flows = []
        for k in tqdm(range(len(self))):
            flows.append(self[k].hamiltonian.flow(t=t*self.lengths[k], **kwargs))
        self.flows = flows
        
    def calcOneTurnMap(self, **kwargs):
        '''
        **kwargs are passed to create_coords routine (e.g. any possible max_power)
        '''
        assert hasattr(self, 'flows'), 'Need to call self.calcFlows first.'
        xiv = create_coords(self.dim, **kwargs)[:self.dim]        
        # N.B. 'reduce' will apply the rightmost function in the given list first, so that e.g.
        # [f0, f1, f2]
        # will be executed as
        # f0(f1(f2(z)))
        # etc.
        # Since in our beamline the first element in the list should be executed first,
        # we have to reverse the order here.
        composition = reduce(f_compose, self.flows[::-1], f_identity)
        self.oneTurnMap = composition(xiv)
        
    def __call__(self, point):
        assert hasattr(self, 'oneTurnMap'), 'Need to call self.calcOneTurnMap first.'
        return [c(point) for c in self.oneTurnMap]

    def magnus(self, power=1, **kwargs):
        '''
        Combine the individual Hamiltonians by means of Magnus series.
        
        Parameters
        ----------
        power: int, optional
            Power in terms of the free parameter (usually the longitudinal coordinate s),
            up to which the magnus series will be exact. 
        
        **kwargs
            Optional keyword arguments passed to lieops.ops.lie.combine routine
        '''            
        hamiltonians = [e.hamiltonian for e in self.sequence]
        self._magnus_series, self._magnus_hamiltonian = combine(*hamiltonians, power=power, 
                                                                lengths=self.lengths, **kwargs)
        return sum(self._magnus_series.values())
        
        
        
        