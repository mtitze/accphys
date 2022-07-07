from functools import reduce
from tqdm.auto import tqdm

from lieops import create_coords, combine
from .tools import f_identity, f_compose

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
        
        self.elements = [e.copy() for e in elements]
        self.ordering = kwargs.get('ordering', list(range(len(elements))))
        
    def __len__(self):
        return len(self.ordering)
    
    def __getitem__(self, key):
        requested_ele_indices = self.ordering[key]
        if type(requested_ele_indices) == list:
            requested_eles = [self.elements[e] for e in requested_ele_indices]
            new_ordering = requested_ele_indices
        else:
            requested_eles = [self.elements[requested_ele_indices]]
            new_ordering = [requested_ele_indices]
        return self.__class__(*requested_eles, ordering=new_ordering)
    
    def __setitem__(self, key, value):
        if value not in self:
            self.ordering[key] = len(self.elements)
            self.elements.append(value)
        else:
            index = self.elements.index(value)
            self.ordering[key] = index
            
    def __iter__(self):
        self._k = 0
        return self
            
    def __next__(self):
        if self._k < len(self):
            self._k += 1
            return self.elements[self.ordering[self._k - 1]]
        else:
            raise StopIteration
            
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
    
    def get_dim(self):
        return self.elements[0].hamiltonian.dim
        
    def setHamiltonians(self, *args, **kwargs):
        '''
        Project the Hamiltonians of the individual elements to specific dimensions.
        '''
        for e in self.elements:
            e.setHamiltonian(*args, **kwargs)
            
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
        self._magnus_series, self._magnus_hamiltonian, self._magnus_forest = combine(*hamiltonians, power=power, 
                                                                                     lengths=self.lengths(), **kwargs)
        return sum(self._magnus_series.values())
        
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
        
    def calcOneTurnMap(self, half=False, **kwargs):
        '''
        Parameters
        ----------
        half: boolean, optional
            If specified, only compute the result for the xi-polynomials.
            
        **kwargs 
            Optional parameters passed to create_coords routine (e.g. any possible max_power)
        '''
        assert hasattr(self, 'flows'), 'Need to call self.calcFlows first.'
        dim0 = self.get_dim()
        xiv = create_coords(dim0, **kwargs)
        if half:
            xiv = xiv[:dim0]
        # N.B. 'reduce' will apply the rightmost function in the given list first, so that e.g.
        # [f0, f1, f2]
        # will be executed as
        # f0(f1(f2(z)))
        # etc.
        # Since in our beamline the first element in the list should be executed first,
        # we have to reverse the order here.
        composition = reduce(f_compose, self.flows[::-1], f_identity)
        self.oneTurnMap = composition(*xiv)
        
    def __call__(self, *point):
        assert hasattr(self, 'oneTurnMap'), 'Need to call self.calcOneTurnMap first.'
        return [c(*point) for c in self.oneTurnMap]
    
    def track(self, *xi0, n_reps: int=1, post=lambda x: x):
        '''
        Perform tracking for a given number of repetitions.

        Parameters
        ----------
        n_reps: int
            The number of repetitions.

        xi0: The start vector xi0.

        post: callable, optional
            An optional function to be applied after bl. 
            Example: We want to compute (A o bl o B)**n at xi0, where A = B**(-1) are maps to normal form.
            Then we need to record for Y0 := B(xi0) the values A(bl(Y0)), A(bl**2(Y0)) etc. Here A = post must
            be inserted.

        Returns
        -------
        list
            A list so that the k'th entry corresponds to the k-th component for the requested turns.
        '''
        point = xi0
        points = []
        for k in range(n_reps):
            point = self(*point)
            points.append(post(point))
        return points
    
    def __str__(self):
        outstr = ''
        for e in self.elements:
            outstr += e.__str__() + '\n'
        return outstr[:-1]

        
        
        
        