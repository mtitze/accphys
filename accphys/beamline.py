import numpy as np

from functools import reduce
from tqdm import tqdm

from lieops import create_coords, combine, lexp
from .tools import f_identity, f_compose

from lieops.solver import heyoka

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
        if type(key) == list:
            requested_ele_indices = [self.ordering[e] for e in key]
        else:
            requested_ele_indices = self.ordering[key]
        if type(requested_ele_indices) != list:
            requested_ele_indices = [requested_ele_indices]
            
        requested_unique_ele_indices = list(np.unique(requested_ele_indices))
        requested_eles = [self.elements[e] for e in requested_unique_ele_indices]
        new_ordering = [requested_unique_ele_indices.index(e) for e in requested_ele_indices] # starts from zero up to length of the unique (new) elements
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
        '''
        Append an element to the current beamline.
        '''
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
    
    def copy(self):
        '''
        Return a copy of the current beamline.
        '''
        assert all([hasattr(e, 'copy') for e in self.elements])
        result = self.__class__(*[e.copy() for e in self.elements])
        # set all remaining fields (including the current ordering)
        for field, value in self.__dict__.items():
            if field != 'elements':
                setattr(result, field, value)
        return result
        
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
        hamiltonians = [e.hamiltonian for e in self][::-1]
        lengths = np.array(self.lengths()[::-1])
        self._magnus_series, self._magnus_hamiltonian, self._magnus_forest = combine(*hamiltonians, power=power, 
                                                                                     lengths=lengths, **kwargs)
        return sum(self._magnus_series.values())
        
    def _calcClassicOneTurnMap(self, half=False, t=-1, *args, **kwargs):
        '''
        Integrate the equations of motion by 'brute-force', namely by calculating the
        flow(s) exp(-:H:) applied to coordinate functions, up to specific orders.
        
        Parameters
        ----------
        half: boolean, optional
            If specified, only compute the result for the xi-polynomials.
            
        **kwargs 
            Optional parameters passed to create_coords routine (e.g. any possible max_power)
        '''
        dim0 = self.get_dim()
        xiv = create_coords(dim0, **kwargs)
        if half:
            xiv = xiv[:dim0]
            
        # create_elemap necessary to construct functions in the loop below
        # This function will be used to compute the flow for each of the unique elements in the lattice
        # (these elements are stored in self.elements).
        def create_elemap(n, **kwargs2):
            e = self.elements[n]
            final_components = lexp(e.hamiltonian, t=t*e.length, **kwargs)(*xiv)
            if 'tol' in kwargs2.keys():
                final_components = [c.drop(kwargs2['tol']) for c in final_components]
            return lambda *z: [c(*z) for c in final_components]
        self._uniqueOneTurnMapOps = []
        for n in tqdm(range(len(self.elements)), disable=kwargs.get('disable_tqdm', False)):
            self._uniqueOneTurnMapOps.append(create_elemap(n, **kwargs))
            
        # Now define the entire one-turn map as composition of the flows we computed.
        # 'reduce' will apply the rightmost function in the given list first, so that e.g.
        # [f0, f1, f2]
        # will be executed as
        # f0(f1(f2(z)))
        # etc.
        # Therefore we have to revert the list below:
        self.oneTurnMapOps = [self._uniqueOneTurnMapOps[k] for k in self.ordering][::-1]
        self.oneTurnMap = reduce(f_compose, self.oneTurnMapOps, f_identity)
        
    def _calcHeyokaOneTurnMap(self, t=1, **kwargs):
        '''
        Integrate the equations of motion using the Heyoka solver, see
        https://bluescarni.github.io/heyoka/index.html
        '''
        # N.B. t=1 here, because t corresponds to the time in the Heyoka integrator. In contrast to
        # the flow, which requires a -1 in the exponent, the signum is already taken care of by
        # integrating the equations of motion.
        self.oneTurnMapOps = []
        for k in tqdm(range(len(self)), disable=kwargs.get('disable_tqdm', False)):
            element_index = self.ordering[k]
            ham = self.elements[element_index].hamiltonian
            length = self.elements[element_index].length
            solver = heyoka(ham, t=t*length, **kwargs)
            self.oneTurnMapOps.append(solver)
        self.oneTurnMapOps = self.oneTurnMapOps[::-1]
        self.oneTurnMap = reduce(f_compose, self.oneTurnMapOps, f_identity)
        
    def calcOneTurnMap(self, *args, method='heyoka', **kwargs):
        if method == 'heyoka':
            self._calcHeyokaOneTurnMap(**kwargs)
        if method == 'classic':
            self._calcClassicOneTurnMap(*args, **kwargs)

    def __call__(self, *point):
        assert hasattr(self, 'oneTurnMap'), 'Need to call self.calcOneTurnMap first.'
        return self.oneTurnMap(*point)
    
    def track(self, *xieta, n_reps: int=1, post=lambda x: x, **kwargs):
        '''
        Perform tracking for a given number of repetitions.

        Parameters
        ----------
        n_reps: int
            The number of repetitions.

        xieta: The start vector (xi, eta).

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
        point = xieta
        points = []
        for k in tqdm(range(n_reps), disable=kwargs.get('disable_tqdm', False)):
            point = self(*point)
            points.append(post(point))
        return points
    
    def __str__(self):
        outstr = ''
        for k in self.ordering:
            outstr += self.elements[k].__str__() + '\n'
        return outstr[:-1]

        
        
        
        