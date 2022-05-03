from functools import reduce
from tqdm.auto import tqdm

from lieops import create_coords, combine

def f_identity(x):
    return x

def f_compose(f, g):
    return lambda x: f(g(x))

class beamline:
    
    def __init__(self, *elements, **kwargs):
        '''
        Class to model an accelerator beamline.
        '''
        # consistency checks
        assert len(elements) > 0
        assert all([hasattr(e, 'hamiltonian') for e in elements])
        dim0 = elements[0].hamiltonian.dim
        assert all([e.hamiltonian.dim == dim0 for e in elements]), 'Dimensions of the individual Hamiltonians differ.'
        
        self.elements = list(elements)
        self.ordering = kwargs.get('ordering', list(range(len(elements))))
        
    def __len__(self):
        return len(self.ordering)
    
    def __getitem__(self, key):
        return self.elements[self.ordering[key]]
    
    def __setitem__(self, key, value):
        if value not in self:
            self.ordering[key] = len(self.elements)
            self.elements.append(value)
        else:
            index = self.elements.index(value)
            self.ordering[key] = index
            
    def index(self, value):
        return self.elements.index(value)
    
    def append(self, value):
        if value not in self.elements:
            index = len(self.elements)
            self.elements.append(value)
        else:
            index = self.elements.index(value)
        self.ordering.append(index)
        
    def lengths(self):
        '''
        Return the lengths of the individual elements.
        '''
        return [e.length for e in self]
        
    def setHamiltonians(self, *args, **kwargs):
        '''
        Project the Hamiltonians of the individual elements to specific dimensions.
        '''
        for e in self.elements:
            e.setHamiltonian(*args, **kwargs)
        
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
        element_flows = [e.hamiltonian.flow(t=t*e.length, **kwargs) for e in self.elements] # compute the flows of the unique elements
        self.flows = [element_flows[j] for j in self.ordering]
        
    def calcOneTurnMap(self, **kwargs):
        '''
        **kwargs are passed to create_coords routine (e.g. any possible max_power)
        '''
        assert hasattr(self, 'flows'), 'Need to call self.calcFlows first.'
        dim0 = self.elements[0].hamiltonian.dim
        xiv = create_coords(dim0, **kwargs)[:dim0]        
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
        hamiltonians = [e.hamiltonian for e in self]
        self._magnus_series, self._magnus_hamiltonian = combine(*hamiltonians, power=power, 
                                                                lengths=[e.length for e in self], **kwargs)
        return sum(self._magnus_series.values())
        
        
        
        