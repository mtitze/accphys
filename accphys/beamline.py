import numpy as np
from tqdm import tqdm
import warnings

from lieops import create_coords, magnus, lexp, poly
from lieops import hadamard2d as hadamard
from lieops.solver import heyoka

from .elements import hard_edge_element

class beamline:
    
    def __init__(self, *elements, **kwargs):
        '''
        Class to model an accelerator beamline.
        
        Parameters
        ----------
        elements
            A series containing elements (or Hamiltonians in form of poly objects). 
            If a poly object has been identified in the series, then a hard-edge model 
            for that element is assumed.
            
        lengths: array-like, optional
            Optional lengths in case that Hamiltonians are given. Will be ignored if elements are provided.
        '''
        # check input and make it consistent
        elements = list(elements) # for item assignment below
        n_elements = len(elements)
        assert n_elements > 0
        for k in range(n_elements):
            # A default hard-edge model will be used. This feature is made for convenience usage.
            if isinstance(elements[k], poly):
                elements[k] = hard_edge_element(elements[k])
            elif isinstance(elements[k], lexp):
                elements[k] = hard_edge_element(elements[k].argument)
        assert all([hasattr(e, 'hamiltonian') for e in elements])
        dim0 = elements[0].hamiltonian.dim
        assert all([e.hamiltonian.dim == dim0 for e in elements]), 'Dimensions of the individual Hamiltonians differ.'
        
        if 'lenghts' in kwargs.keys():
            for k in range(n_elements):
                elements[k].length = kwargs['lengths'][k]
            
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
            return self.elements[requested_ele_indices]
        else:
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
    
    def append_element(self, value):
        '''
        Append an element to the current beamline.
        '''
        if value not in self.elements:
            index = len(self.elements)
            self.elements.append(value)
        else:
            index = self.elements.index(value)
        self.ordering.append(index)
        
    def __add__(self, other):
        result = self.copy()
        if isinstance(other, type(self)):
            for e in other:
                result.append_element(e)
        elif isinstance(other, type(self[0])): # a single element
            result.append_element(other)
        else:
            raise TypeError(f'Addition between objects of types {type(self)} and {type(other)} not supported.')            
        return result
            
    def __radd__(self, other):
        if isinstance(other, type(self)):
            return other.__add__(self)
        elif isinstance(other, type(self[0])): # a single element
            result = self.__class__(other)
            return result + self
        else:
            raise TypeError(f'Addition between objects of types {type(other)} and {type(self)} not supported.')
            
    def __mul__(self, other):
        assert type(other) == int, 'Only integer values allowed.'
        result = self.copy()
        result.ordering = result.ordering*other
        return result
    
    def __rmul__(self, other):
        return self*other
        
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
        result.ordering = [e for e in self.ordering]
        return result
        
    def setHamiltonians(self, *args, **kwargs):
        '''
        Project the Hamiltonians of the individual elements to specific dimensions.
        '''
        for e in self.elements:
            e.setHamiltonian(*args, **kwargs)
            
    def magnus(self, power=1, bch_sign=-1, **kwargs):
        '''
        Combine the individual Hamiltonians by means of Magnus series.
        
        Parameters
        ----------
        power: int, optional
            Power in terms of the free parameter (usually the longitudinal coordinate s),
            up to which the magnus series will be exact. 
        
        **kwargs
            Optional keyword arguments passed to lieops.core.combine.magnus routine
        '''            
        hamiltonians = [e.hamiltonian*bch_sign for e in self]
        lengths = np.array(self.lengths())
        self._magnus_series, self._magnus_hamiltonian, self._magnus_forest = magnus(*hamiltonians, power=power, 
                                                                                     lengths=lengths, **kwargs)
        return self.__class__(hard_edge_element(sum(self._magnus_series.values())*bch_sign, length=1))
    
    def breakdown(self, check=False):
        '''
        Obtain a list of unique hamiltonians and their relation to the current beamline elements.
        '''
        hh = [] # a helper list to gather the unique hamiltonians
        uniques = []
        for k in range(len(self.elements)):
            element = self.elements[k]
            ham = element.hamiltonian
            length = element.length
            if ham not in hh:
                hh.append(ham)
                uniques.append({'hamiltonian': ham, 'length': [length], 'element_index': [k]})
            else:
                index = hh.index(ham)
                uniques[index]['length'].append(length)
                uniques[index]['element_index'].append(k)
                
        if check:
            # consistency check if there are 'different' elements having the same Hamiltonian *and* the same length:
            k = 0
            for u in uniques:
                if len(np.unique(u['length'])) < len(u['length']):
                    warnings.warn(f'Entry {k} has non-unique lengths.')
                k += 1
        
        return uniques
        
    def _calcOneTurnMap_bruteforce(self, half=False, t=-1, *args, **kwargs):
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
            final_components = lexp(e.hamiltonian*e.length, **kwargs)(*xiv, t=t) # TODO: sign
            if 'tol' in kwargs2.keys():
                final_components = [c.above(kwargs2['tol']) for c in final_components]
            return lambda *z: [c(*z) for c in final_components] # z: point of interest
        self._uniqueOneTurnMapOps = []
        for n in tqdm(range(len(self.elements)), disable=kwargs.get('disable_tqdm', False)):
            self._uniqueOneTurnMapOps.append(create_elemap(n, **kwargs))
        self.oneTurnMapOps = [self._uniqueOneTurnMapOps[k] for k in self.ordering]
            
    def _calcOneTurnMap_heyoka(self, t=1, **kwargs):
        '''
        Using the Heyoka solver one-by-one on each element. This may become very slow for large beamlines,
        but may be useful for the analysis/diagonsis of individual elements.
        
        Further details see
        https://bluescarni.github.io/heyoka/index.html
        '''
        self.oneTurnMapOps = []
        for k in tqdm(range(len(self)), disable=kwargs.get('disable_tqdm', False)):
            element_index = self.ordering[k]
            ham = self.elements[element_index].hamiltonian
            length = self.elements[element_index].length
            solver = heyoka(ham, t=t*length, **kwargs)
            self.oneTurnMapOps.append(solver)
            
    def _calcOneTurnMap_channell(self, t=1, **kwargs):
        '''
        Using Yoshida split & Channell's symplectic integrator.
        '''
        # create_elemap necessary to construct functions in the loop below
        # This function will be used to compute the flow for each of the unique elements in the lattice
        # (these elements are stored in self.elements).
        def create_elemap(n):
            e = self.elements[n]
            ele_map = lexp(-e.hamiltonian*e.length) # TODO: sign; note that any sign in **kwargs will be recognized in ele_map.calcFlow in the next line.
            ele_map.calcFlow(method='channell', **kwargs)
            return ele_map.flow
        
        self._uniqueOneTurnMapOps = []
        for n in tqdm(range(len(self.elements)), disable=kwargs.get('disable_tqdm', False)):
            self._uniqueOneTurnMapOps.append(create_elemap(n))
        self.oneTurnMapOps = [self._uniqueOneTurnMapOps[k] for k in self.ordering]
        
    def calcOneTurnMap(self, *args, method='bruteforce', **kwargs):
        if method == 'bruteforce':
            self._calcOneTurnMap_bruteforce(*args, **kwargs)
        elif method == 'channell':
            self._calcOneTurnMap_channell(*args, **kwargs)
        elif method == 'heyoka':
            self._calcOneTurnMap_heyoka(**kwargs)
        else:
            raise RuntimeError(f'Method {method} not recognized.')
        self._oneTurnMapMethod = method

    def __call__(self, *point):
        self.out = []
        assert hasattr(self, 'oneTurnMapOps'), 'Call self.calcOneTurnMap first.'
        for m in self.oneTurnMapOps:
            point = m(*point)
            self.out.append(point)
        return point
    
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

    def split(self, *args, **kwargs):
        '''
        Split beamline elements according to a given splitting scheme.
        
        Parameters
        ----------
        **kwargs
            Keyworded parameters passed to the split routine of each beamline element.
        
        Returns
        -------
        beamline
            A beamline containing the splitted elements.
        '''
        # split the elements of the current beamline
        new_elements = []
        new_ordering_indices = []
        ordering_index = 0
        
        for e in self.elements:            
            esplit, ordering = e.split(*args, return_scheme_ordering=True, **kwargs)
            
            n_unique_elements = max(ordering) + 1
            unique_element_indices = list(range(n_unique_elements))
            new_elements += [esplit[ordering.index(k)] for k in unique_element_indices] # take the first occurences
            
            ordering_element = (np.array(ordering) + ordering_index).tolist()
            ordering_index += n_unique_elements
            
            new_ordering_indices.append(ordering_element)
                                        
        # construct the new ordering
        new_ordering = []
        for k in self.ordering:
            new_ordering += new_ordering_indices[k]
            
        assert max(new_ordering) + 1 == len(new_elements) # consistency check
        return self.__class__(*new_elements, ordering=new_ordering)
            
    def hadamard(self, keys, **kwargs):
        '''
        Rearrange the Hamiltonians of the elements in the current beamline according to given keys.
        Further details see lieops 'hadamard' routine.
        
        Returns
        -------
        beamline
            A beamline of hard_edge_element(s) corresponding to the output of hadamard.
        '''
        t = kwargs.get('t', 1)
        hamiltonians = [e.hamiltonian*e.length*t for e in self][::-1] # the leftmost operator belongs to the element at the end of the beamline; TODO: sign

        g1, g2, g2_all = hadamard(*hamiltonians, keys=keys, **kwargs) # a higher power may be necessary here ...
        new_elements = [hard_edge_element(h1, length=1) for h1 in g1] + [hard_edge_element(h2, length=1) for h2 in g2] 
        out = self.__class__(*new_elements[::-1])
        out._hadamard_trail = g2_all[::-1]
        return out
