import numpy as np
from tqdm import tqdm
import warnings

from lieops import create_coords, magnus, lexp, poly
from lieops.core import dragtfinn
from lieops.core.hadamard import reshuffle2d as reshuffle
from lieops.core.forest import fnf
from lieops.core.tools import poly2vec, tpsa
from lieops.linalg.checks import symplecticity

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
            
    def calcOneTurnMap(self, *args, **kwargs):
        '''
        Calculate the flow of each element explicitly (i.e. call calcFlow for each element operator).        
        '''
        for e in tqdm(self.elements, disable=kwargs.get('disable_tqdm', False)):
            e.operator.calcFlow(*args, **kwargs)
            
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
        # bch_sign = -1 is used here because the arguments are the negatives of the Hamiltonians (times their lengths) and 
        # the magnus routine is working with the pure arguments & the lengths.
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

    def __call__(self, *point, **kwargs):
        self.out = []
        for e in self:
            point = e(*point, **kwargs)
            self.out.append(point)
        return point
    
    def arguments(self):
        '''
        Return the arguments f in the individual Lie operators exp(:f:) of the beamline.
        Note that these arguments differ from the Hamiltonians by -1 and the length.
        
        The order of the return list will agree with the order of the elements in the lattice.
        '''
        assert all([hasattr(e, 'operator') for e in self])
        return [e.operator.argument for e in self]
    
    def operators(self):
        '''
        Return the operators exp(:f:) of the beamline.
        
        The order of the return list will agree with the order of the elements in the lattice.
        '''
        assert all([hasattr(e, 'operator') for e in self])
        return [e.operator for e in self]
    
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
            
    def reshuffle(self, keys, **kwargs):
        '''
        Rearrange the Hamiltonians of the elements in the current beamline according to given keys.
        
        Returns
        -------
        beamline
            A beamline of hard_edge_element(s) according to lieops.core.hadamard.reshuffle2d.
        '''
        hamiltonians = [e.operator.argument for e in self][::-1] # the leftmost operator belongs to the element at the end of the beamline (+)
        g1, g2, g2_all = reshuffle(*hamiltonians, keys=keys, **kwargs) # a higher power may be necessary here ...
        new_elements = [hard_edge_element(lexp(h1), length=1) for h1 in g1] + [hard_edge_element(lexp(h2), length=1) for h2 in g2] # h1 and h2 are considered to be the full arguments of the operators, so we initiate the elements with lexp objects.
        out = self.__class__(*new_elements) # no reverse: The first element in new_elements is acting last, but on lie-polynomials. On points it is acting *first* due to the pull-back property on numbers. So the first element also corresponds to the first element in the new lattice.
        out._reshuffle_trail = g2_all
        return out
    
    def tpsa(self, *position, tol=1e-14, **kwargs):
        '''
        Pass n-jets through the flow functions of the individual elements.
        
        The results are stored in self._tpsa*. 
        
        Parameters
        ----------
        *position: float or array, optional
            An optional point of reference. By default the position will be the origin.
            
        order: int
            The number of derivatives we want to take into account.
            
        tol: float, optional
            A tolerance to check (if > 0) if the resulting map is indeed symplectic.
            It is recommended to perform this check to avoid errors in places where
            the one-turn map might be used.
            
        **kwargs
            Optional keyworded arguments passed to lieops.core.tools.tpsa (and therefore the underlying
            operators of this beamline).
        '''
        _ = kwargs.setdefault('position', position)
        tpsa_out = tpsa(*self.operators(), **kwargs)
        taylor_map = tpsa_out['taylor_map']
        if tol > 0: # check if map is symplectic; it is recommended to do this check here to avoid errors in routines which use the Taylor map.
            R = np.array([poly2vec(e.homogeneous_part(1)).tolist() for e in taylor_map])
            check, message = symplecticity(R, tol=tol, warn=kwargs.get('warn', True))
        self._tpsa = tpsa_out
        return tpsa_out
    
    def _memCheck_tpsa(self, *position, order: int, tol_checks=0, **kwargs):
        '''
        Check whether it is necessary to perform a TPSA calculation by
        looking at the position, the order and other input parameters.
        '''
        compute_tpsa = True
        if hasattr(self, '_tpsa'):
            # Check if the input position and the order agrees with the one already stored. If not, re-do the TPSA calculation.
            tpsa_order = self._tpsa['DA'].order
            tpsa_position = self._tpsa['position']
            compute_tpsa = not (tpsa_order >= order) or not all([tpsa_position[k] == position[k] for k in range(self.get_dim()*2)])
            # Compute the input parameters for TPSA and check if that has changed:
            compute_tpsa = compute_tpsa or self._tpsa['input'] != kwargs
        if compute_tpsa:
            _ = self.tpsa(*position, order=order, tol=tol_checks, **kwargs)

    def dragtfinn(self, *position, order: int, **kwargs):
        '''
        Compute the Dragt/Finn factorization of the current Taylor map (self._tpsa_taylor_map) of the lattice.
        
        Parameters
        ----------
        *position: coordinate(s)
            The point of interest at which the Taylor map should be considered.
            
        order: int
            The maximal order of terms in the Dragt/Finn factorization. This value should preferably
            take into account the order of the given Taylor-map.
            
        tol_checks: float, optional
            A parameter to perform certain consistency checks.
            
        **kwargs
            Arguments passed to dragtfinn routine. In particular one should
            provide flow calculation parameters to the underlying Lie operators.
            
        Returns
        -------
        beamline
            An object of type self.__class__ corresponding to a beamline in which the elements are given by
            the result of the Factorization. Note that the original lengths of the elements will (and can) not be preserved.
        '''
        # I) Check whether it is necessary to perform a TPSA calculation prior to dragtfinn
        tpsa_order = kwargs.pop('tpsa_order', order)
        # Determine the TPSA input (by separating the dragtfinn input from the pure flow input) 
        tpsa_input = kwargs.copy()
        for key in ['offset', 'pos2', 'comb2', 'tol', 'tol_checks', 'force_order']:
            _ = tpsa_input.pop(key, None)
        self._memCheck_tpsa(*position, order=tpsa_order, **tpsa_input)
            
        # II) Perform the Dragt/Finn factorization
        _ = kwargs.setdefault('offset', self._tpsa['position'])
        df = dragtfinn(*self._tpsa['taylor_map'], order=order, **kwargs)
        return self.__class__(*[lexp(f) for f in df]) # use lexp objects here so that the elements in df are properly recognized as the full arguments of the operators. Note also that, by construction of the 'dragtfinn' routine, the first element in df needs to be executed first on the coordinates, so it has to stay at the beginning of the beamline.
    
    def normalform(self, *position, order: int, **kwargs):
        '''
        Perform a normal form analysis of the beamline, using lieops.core.forest.fnf, see
        Ref. [1] or the lieops routine for details.
        
        Note that (currently) only positions at zero are supported.
        
        Reference(s)
        ------------
        [1] E. Forest: "Beam Dynamics - A New Attitude and Framework", harwood academic publishers (1998).
        '''
        # I) Check whether it is necessary to perform a TPSA calculation prior to fnf;
        # determine the TPSA input (by separating the fnf input from the pure flow input) 
        tpsa_order = kwargs.pop('tpsa_order', order)
        tpsa_input = kwargs.copy()
        for key in ['bch_order', 'mode', 'offset', 'pos2', 'comb2', 'tol', 'tol_checks', 'force_order']:
            _ = tpsa_input.pop(key, None)
        self._memCheck_tpsa(*position, order=tpsa_order, **tpsa_input)
        
        # II) Perform the normal form analysis
        return fnf(*self._tpsa['taylor_map'], order=order, **kwargs)
    