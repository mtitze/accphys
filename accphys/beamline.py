import numpy as np
from tqdm import tqdm
import warnings

from njet import derive

from lieops import create_coords, magnus, lexp, poly
from lieops import hadamard2d as hadamard
from lieops.core import dragtfinn

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
            if isinstance(elements[k], poly) or isinstance(elements[k], lexp):
                elements[k] = hard_edge_element(elements[k])
            # else: elements[k] = elements[k]
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
            
    def magnus(self, order=1, bch_sign=-1, **kwargs):
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
        self._magnus_series, self._magnus_hamiltonian, self._magnus_forest = magnus(*hamiltonians, order=order, 
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
        
    def calcOneTurnMap(self, **kwargs):
        '''
        Integrate the equations of motion by 'brute-force', namely by calculating the
        flow(s) exp(-:H:) applied to coordinate functions, up to specific orders.
        
        Parameters
        ----------
        method: str, optional
            The method which should be used for the elements in the beamline.
            
        **kwargs 
            Optional parameters passed to
            accphys.elements.hard_edge_element.calcOneTurnMap 
            routine (e.g. any possible max_power)
        '''
        for n in tqdm(range(len(self.elements)), disable=kwargs.get('disable_tqdm', False)):
            self.elements[n].calcOneTurnMap(**kwargs)
        
    def __call__(self, *point, **kwargs):
        self.out = []
        assert all([hasattr(e, 'oneTurnMap') for e in self.elements]), 'Some elements require calculation of their one-turn map.'
        for e in self:
            point = e(*point, **kwargs)
            self.out.append(point)
        return point
    
    def track(self, *xieta, n_reps: int=1, post=lambda *x: x, real=False,
              output_format='default', **kwargs):
        '''
        Perform tracking for a given number of repetitions.

        Parameters
        ----------
        n_reps: int
            The number of repetitions.

        xieta: The start vector (xi, eta).

        post: callable, optional
            An optional function to be applied after each tracking.
            Example: 
            Let 'bl' denote the map of the current beamline. 
            We want to compute (A o bl o B)**n at xi0, where A = B**(-1) are maps to normal form.
            Then we need to record for Y0 := B(xi0) the values A(bl(Y0)), A(bl**2(Y0)) etc. 
            A = post.
            
        real: boolean, optional
            If True, assume the input is given in terms of standard q-p coordinates.
            
        output_format: str, optional
            Determine the format of the output as follows:
            default/list: A single object will be returned, which corresponds to a list of data, 
                          where the k-th element corresponds to the output of turn k (default).
            transposed/coords: dim objects will be returned (one for each coordinate), each containing 
                               the computed points for the requested turns.

        Returns
        -------
        list or series of lists
            Output according to 'output_format'.
        '''        
        # Some input consistency checks
        dim2 = len(xieta)
        assert dim2%2 == 0
        assert dim2//2 == self.get_dim()
        output_formats = ['default', 'list', 'coords', 'transposed'] # supported output formats
        assert output_format in output_formats, f"Output format '{output_format}' not recognized."
    
        # Prepare input and post-function in case of 'real-valued' data
        if real:
            sqrt2 = float(np.sqrt(2))
            dim = dim2//2
            q0 = xieta[:dim]
            p0 = xieta[dim:]
            xi0 = [(q0[k] + p0[k]*1j)/sqrt2 for k in range(dim)]
            eta0 = [(q0[k] - p0[k]*1j)/sqrt2 for k in range(dim)]
            xieta = xi0 + eta0

            def post_real(*z):
                '''
                Map complex xi/eta-coordinates to real q/p-coordinates,
                then apply the user-given 'post'-map.
                '''
                xif = z[:dim]
                etaf = z[dim:]
                qfpf = [(xif[k] + etaf[k])/sqrt2 for k in range(dim)] + \
                       [(xif[k] - etaf[k])/sqrt2/1j for k in range(dim)]
                return post(*qfpf)
                
            post_used = post_real
        else:
            post_used = post
            
        point = xieta
        points = []
        for k in tqdm(range(n_reps), disable=kwargs.get('disable_tqdm', False)):
            point = self(*point, **kwargs)
            points.append(post_used(*point))
            
        if output_format in ['default', 'list']:
            return points
        elif output_format in ['coords', 'transposed']:
            return (*[[points[k][j] for k in range(n_reps)] for j in range(dim2)],)
    
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
        hamiltonians = [e.hamiltonian*e.length for e in self][::-1] # the leftmost operator belongs to the element at the end of the beamline

        g1, g2, g2_all = hadamard(*hamiltonians, keys=keys, **kwargs) # a higher power may be necessary here ...
        new_elements = [hard_edge_element(h1, length=1) for h1 in g1] + [hard_edge_element(h2, length=1) for h2 in g2] 
        out = self.__class__(*new_elements[::-1]) # again revert the order, because the last element in 'new_elements' will be executed first.
        out._hadamard_trail = g2_all[::-1]
        return out
    
    def tpsa(self, *position, order: int):
        '''
        Pass n-jets through the flow functions of the individual elements.
        
        Parameters
        ----------
        *position: float or array, optional
            An optional point of reference. By default the position will be the origin.
            
        order: int
            The number of derivatives we want to take into account.
        '''
        dim = self.get_dim()
        n_args = dim*2
        self._tpsa = derive(self, n_args=n_args, order=order)
        if len(position) == 0:
            position = (0,)*n_args
        expansion = self._tpsa(*position, mult_prm=True, mult_drv=False) # N.B. the plain jet output is stored in self._tpsa._evaluation. From here one can use ".get_taylor_coefficients" with other parameters -- if desired -- or re-use the jets for further processing.
        max_power = max([e.hamiltonian.max_power for e in self.elements])
        taylor_map = [poly(values=e, dim=dim, max_power=max_power) for e in expansion]
        self._tpsa_position = position
        self._tpsa_taylor_map = taylor_map
        return taylor_map

    def dragtfinn(self, *position, order: int, **kwargs):
        '''
        Pass n-jets through the lattice at a point of interest. Then return the symplectic approximation (Dragt/Finn factorization)
        of the map near that point.
        '''
        # I) Check whether it is necessary to perform a TPSA calculation in advance
        tpsa_order = kwargs.pop('tpsa_order', order)
        compute_tpsa = True
        if hasattr(self, '_tpsa'):
            # Check if the input position and the order agrees with the one already stored. If not, re-do the TPSA calculation.
            compute_tpsa = not (self._tpsa.order >= tpsa_order) or not all([self._tpsa_position[k] == position[k] for k in range(self.get_dim()*2)])
        if compute_tpsa:
            _ = self.tpsa(*position, order=tpsa_order)
            
        # II) Perform the Dragt/Finn factorization
        df = dragtfinn(*self._tpsa_taylor_map, offset=position, order=order, **kwargs)
        df = [-f for f in df] # the minus signs are in place to compensate the one made in __call__; TODO: sign ...
        return self.__class__(*df, offset=position, **kwargs)
    