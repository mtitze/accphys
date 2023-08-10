import numpy as np
from tqdm import tqdm
import warnings
from copy import deepcopy

from njet.common import check_zero

from lieops import create_coords, magnus, lexp, poly
from lieops.core import dragtfinn
from lieops.core.hadamard import reshuffle2d as reshuffle
from lieops.core.forest import fnf
from lieops.core.tools import poly2vec, tpsa, symcheck, taylor_map
from lieops.linalg.matrix import emat

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
        for k in range(n_elements):
            # A default hard-edge model will be used. This feature is made for convenience usage.
            if isinstance(elements[k], poly) or isinstance(elements[k], lexp):
                elements[k] = hard_edge_element(elements[k])
            # else: elements[k] = elements[k]
        assert all([hasattr(e, 'hamiltonian') for e in elements])
        if len(elements) > 0:
            dim0 = elements[0].hamiltonian.dim
            assert all([e.hamiltonian.dim == dim0 for e in elements]), 'Dimensions of the individual Hamiltonians differ.'
            
        self.elements = [deepcopy(e) for e in elements] # deepcopy required here to also copy any possible existing flow function(s) attached to operators
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
        '''
        Return the element number at a given position in the beamline chain (similar as .index for lists)
        '''
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
        if other > 0:
            result = self.copy()
        elif other < 0:
            result = self.__class__(*[lexp(-c) for c in self.arguments()][::-1])
        else: # other == 0; return a beamline consisting of a single element (the identity)
            return self.__class__(poly(values={}, dim=self.dim())) # dim needs to be set in poly here, in order to provide the identity operator.
        result.ordering = result.ordering*max([1, abs(other)])
        return result
    
    def __rmul__(self, other):
        return self*other
    
    def __neg__(self):
        return self*-1
    
    def __sub__(self, other):
        return self + other*-1
    
    def __rsub__(self, other):
        return other + self*-1
        
    def lengths(self, default=1):
        '''
        Return the lengths of the individual elements. 
        
        Parameters
        ----------
        default: float, optional
            If an element has no length attribute, its length is set to this value.

        Returns
        -------
        list
            A list of floats, where entry k corresponds to the length of the k-th element.
        '''
        return [getattr(e, 'length', default) for e in self]
    
    def positions(self, default=1):
        '''
        Return the positions of the individual elements along the beamline.
        The first element starts at position zero. 
        
        Parameters
        ----------
        default: float, optional
            The positions are inferred from the lengths of the individual elements. If an element has no length
            attribute, its length is set to this value.

        Returns
        -------
        ndarray
            A set of positions, where entry k denotes the start position of the k-th element.            
        '''
        return np.cumsum([0] + [getattr(e, 'length', default) for e in self][:-1])
    
    def dim(self):
        return self.elements[0].hamiltonian.dim
    
    def apply(self, *args, **kwargs):
        for e in self.elements:
            e.apply(*args, **kwargs) 
    
    def copy(self):
        '''
        Return a copy of the current beamline.
        '''
        return deepcopy(self)
        
    def project(self, *args, **kwargs):
        '''
        Project the Hamiltonians of the individual elements to specific dimensions.
        '''
        return self.__class__(*[e.project(*args, **kwargs) for e in self.elements], ordering=self.ordering)
            
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
        for e in tqdm(self, disable=kwargs.get('disable_tqdm', True)):
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
    
    def track(self, *xieta, turns: int=1, post=lambda *x: x, real=False,
              output_format='default', record=False, **kwargs):
        '''
        Perform tracking for a given number of repetitions.

        Parameters
        ----------
        turns: int
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
            
        record: boolean, optional
            If True, also store all values from the various elements of the beamline.
            
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
        assert dim2//2 == self.dim()
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
        all_out = []
        for k in tqdm(range(turns), disable=kwargs.get('disable_tqdm', False)):
            point = self(*point, **kwargs)
            points.append(post_used(*point))
            if record:
                all_out.append([post_used(*pt) for pt in self.out])
            
        self.out_track = all_out
            
        if output_format in ['default', 'list']:
            return points
        elif output_format in ['coords', 'transposed']:
            return (*[[points[k][j] for k in range(turns)] for j in range(dim2)],)
    
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
        g1, g2, g2_all = reshuffle(*[e.operator.argument for e in self], keys=keys, **kwargs) # a higher power may be necessary here ...
        new_elements = [hard_edge_element(lexp(h1), length=1) for h1 in g1] + [hard_edge_element(lexp(h2), length=1) for h2 in g2] # h1 and h2 are considered to be the full arguments of the operators, so we initiate the elements with lexp objects.
        out = self.__class__(*new_elements) # no reverse: The first element in new_elements is acting last, but on lie-polynomials. On points it is acting *first* due to the pull-back property on numbers. So the first element also corresponds to the first element in the new lattice.
        out._reshuffle_trail = g2_all
        return out
    
    def _tpsa_memcheck(self, *position, force=False, **kwargs):
        '''
        Check if (re-)calculation of TPSA is necessary.
        '''
        assert 'order' in kwargs.keys()
        compute_tpsa = True
        if hasattr(self, '_tpsa_input') and not force:
            stored_input = self._tpsa_input      
            compute_tpsa = stored_input['order'] < kwargs['order']

            stored_position = stored_input['position']
            if len(stored_position) > 0 and len(position) > 0:
                try:
                    compute_tpsa = compute_tpsa or not all([check_zero(stored_position[k] - position[k]) for k in range(self.dim()*2)])
                except:
                    # We may drop in an exception if stored_position[k] and position[k] have different shapes. In this case: 
                    compute_tpsa = True
            else:
                # stored_position is [] or position is [].
                # This condition may happen if the user requested to return a 'derive' or 'cderive' object in TPSA without
                # any point evaluation (and so without any position request).
                compute_tpsa = compute_tpsa or not (len(stored_position) == 0 and len(position) == 0) # if stored_position is None and position is None, then both are 'equal' and so TPSA needs not be calculated by this condition. Otherwise TPSA needs to be (re)calculated.
            remaining_stored_input = {a: b for a, b in stored_input.items() if a != 'order' or a != 'position'}
            compute_tpsa = compute_tpsa or not kwargs.items() <= remaining_stored_input.items()
        return compute_tpsa
    
    def tpsa(self, *position, force=False, **kwargs):
        '''
        Pass n-jets through the flow functions of the individual elements.
        
        The results are stored in self._tpsa*. 
        
        Parameters
        ----------
        *position: float or array, optional
            An optional point of reference.
            
        order: int
            The number of derivatives we want to take into account.
            
        tol: float, optional
            A tolerance to check (if > 0) if the resulting map is indeed symplectic.
            It is recommended to perform this check to avoid errors in places where
            the one-turn map might be used.
        
        force: boolean, optional
            Bypass memory check and force (re-)calculation.
            
        **kwargs
            Optional keyworded arguments passed to lieops.core.tools.tpsa (and therefore the underlying
            operators of this beamline).
            
        Returns
        -------
        dict
            The output of lieops.core.tools.tpsa; will also be stored in self._tpsa
        '''
        if len(position) == 0:
            position = (0,)*self.dim()*2

        assert 'order' in kwargs.keys(), "'order' key required for TPSA calculation."
        if self._tpsa_memcheck(*position, force=force, **kwargs):
            if len(position) > 0:
                kwargs['position'] = position
            self._tpsa = tpsa(*[e.operator for e in self.elements], ordering=self.ordering, **kwargs)
            
        # store input for later use:
        self._tpsa_input = kwargs.copy()
        self._tpsa_input['position'] = position
        return self._tpsa
        
    def taylor_map(self, *position, tol=1e-14, **kwargs):
        tpsa_out = self.tpsa(*position, **kwargs) # TPSA inclues a memory check by default.
        assert hasattr(tpsa_out, '_evaluation'), 'TPSA jet-evaluation at specific point required.'
        
        dim = self.dim()
        default_max_power = min([e.operator.argument.max_power for e in self.elements])
        self._taylor_map = taylor_map(*tpsa_out._evaluation, dim=dim, max_power=kwargs.get('max_power', default_max_power))
        
        if tol > 0 and kwargs.get('warn', True): 
            # Check if Taylor map is symplectic. It is recommended to do this check here to avoid errors in routines which use the Taylor map.
            check_results = symcheck(self._taylor_map, tol=tol, warn=kwargs.get('warn', False))
            if len(check_results) > 0:
                min_order = min(list(check_results.keys()))
                error = check_results[min_order]
                warnings.warn(f'Taylor map not symplectic for order >= {min_order}: {error} (tol: {tol})')
        return self._taylor_map
    
    def dragtfinn(self, **kwargs):
        '''
        Compute the Dragt/Finn factorization of the current Taylor map (self._tpsa_taylor_map) of the lattice.
        
        Parameters
        ----------
        **kwargs
            Arguments passed to lieops.core.dragt.dragtfinn routine. In particular one should
            provide flow calculation parameters to the underlying Lie operators.
            
        Returns
        -------
        beamline
            An object of type self.__class__ corresponding to a beamline in which the elements are given by
            the result of the Factorization. Note that the original lengths of the elements will (and can) not be preserved.
        '''
        assert hasattr(self, '_taylor_map'), 'Taylor map calculation required in advance.'
        _ = kwargs.setdefault('offset', self._tpsa_input['position'])
        df = dragtfinn(*self._taylor_map, **kwargs)
        return self.__class__(*[lexp(f) for f in df]) # use lexp objects here so that the elements in df are properly recognized as the full arguments of the operators. Note also that, by construction of the 'dragtfinn' routine, the first element in df needs to be executed first on the coordinates, so it has to stay at the beginning of the beamline.
    
    def _normalform(self, tmap, **kwargs):
        r'''
        Internal routine used in self.normalform and self.optics; 
        call lieops.core.forest.fnf with a given Taylor map.
        
        Returns
        -------
        dict
            A dictionary containing the output X of lieops.core.forest.fnf, as well as the following items:
                 normalbl: A beamline object consisting of a single "hard-edge" element, representing
                           the normalized (and commuting) terms of the current beamline.
                        N: A beamline object consisting of those terms mapping the current beamline to
                           its normal form.
                       Ni: The inverse of N.
                           Hereby it holds:
                           1) normalbl = N + self + Ni
                           2) self = Ni + normalbl + N
            courantsnyder: A dictionary containing the Courant/Snyder linear optics functions alpha, beta and gamma.
                           Hereby a key of the form alpha<ij> maps to the values of the alpha-function relative to the
                           coordinates z_i and z_{j + dim}, while the beta<i> and gamma<i> keys map to the respective
                           z_{i + dim}**2 and z_i**2 values.
        '''
        fnfdict = fnf(*tmap, **kwargs)
        
        # Add some useful keys
        nf = fnfdict['normalform']
        if len(nf) > 0:
            fnfdict['normalbl'] = self.__class__(lexp(sum(nf)))
        else:
            warnings.warn('Normal form appears to be the identity transformation.')
            fnfdict['normalbl'] = self*0
        fnfdict['N'] = self.__class__(*[lexp(c) for c in fnfdict['chi'][::-1]])
        fnfdict['Ni'] = fnfdict['N']*-1
        
        if 'bnfout' in fnfdict.keys():
            # Compute the linear optics function(s) alpha, beta, gamma.
            #
            # These functions can be derived by noting that the matrix S transforms the ordinary (q, p)-coordinates
            # to normalized (q, p)-coordinates. Therefore, with z := (q, p), w := S@z and
            # 
            #        / gamma   alpha \
            #  G := |                |
            #       \ alpha    beta /
            #
            # we have:
            #
            # w^{tr}@w = z^{tr}@G@z, i.e.
            # z^{tr}@G@z = z^(tr)(S@z)^(tr)@S@z = z^{tr}@S^{tr}@S@z,
            #
            # so that
            #
            # G = S^{tr}@S
            #
            # From this equation we can deduce the optics functions:
            dim = self.dim()
            S = emat(fnfdict['bnfout']['nfdict']['S'])
            courant_snyder = (S.transpose().conjugate()@S).matrix.real # 'conjugate' is used here to remove a possible minus sign in the negative definite case (i.e. if tunes have the opposite sign).
            csd = {}
            for k in range(dim):
                csd[f'gamma{k}'] = courant_snyder[k, k, ...]
                csd[f'beta{k}'] = courant_snyder[k + dim, k + dim, ...]
                for l in range(dim):
                    csd[f'alpha{k}{l}'] = courant_snyder[k, l + dim, ...]
            fnfdict['courantsnyder'] = csd

            # tunes
            fnfdict['tune'] = np.array(fnfdict['bnfout']['mu'])/2/np.pi
        return fnfdict
    
    def normalform(self, **kwargs):
        '''
        Perform a normal form analysis of the beamline, using lieops.core.forest.fnf, see
        Ref. [1] or the lieops routine for details.
                
        Parameters
        ----------
        order: int, optional
            The degree of the normal form procedure. For example: A degree 4 will yield normal form
            terms in the action up and including 2nd order (action**2 = xi**2*eta**2).
               
        **kwargs
            Additional keyworded arguments passed to lieops.core.forest.fnf
            
        Returns
        -------
        dict
            A dictionary containing the output of self._normalform.
        
        Reference(s)
        ------------
        [1] E. Forest: "Beam Dynamics - A New Attitude and Framework", harwood academic publishers (1998).
        '''
        assert hasattr(self, '_taylor_map'), 'Taylor map calculation required in advance.'
        return self._normalform(self._taylor_map, **kwargs)
        
    def cycle(self, *point, order: int, **kwargs):
        '''
        Invoke a TPSA calculation for the entire chain, assuming cyclic conditions.
        
        By default, an njet.extras.cderive object will be returned. This object is also
        stored in self._cycle.
        
        Parameters
        ----------
        *point
            Point at lattice start at which to evaluate the TPSA along the beamline.
            
        order: int, optional
            The order of the TPSA calculation.
            
        **kwargs
            Keyworded parameters used for the flow calculation of the underlying elements of the beamline.
        '''
        _ = self.tpsa(*point, order=order, mode='chain', **kwargs)
        _ = kwargs.setdefault('outf', 0)
        self._cycle = self._tpsa.cycle(*self._tpsa_input['position'], **kwargs)
        return self._cycle
    
    def optics(self, **kwargs):
        '''
        Get the detuning and driving terms (map to normal form; phase space distortion) along the beamline.
        The resolution depends on the given elements in the beamline.
                
        Parameters
        ----------
        max_power: int, optional
            max_power of Taylor map to be used.
        
        warn: boolean, optional
            Display warning in case an internal check went wrong (default: False).
        
        **kwargs
            Further keyworded parameters given to lieops.core.forest.fnf
            
        Returns
        -------
        dict
            A dictionary containing the output of self._normalform.
        '''
        # Input parameter handling
        disable_tqdm = kwargs.get('disable_tqdm', False)
        kwargs['disable_tqdm'] = True # Turn off progress bar for the inner loops
        _ = kwargs.setdefault('warn', False)
        
        assert hasattr(self, '_cycle'), 'TPSA cycle calculation has to be performed in advance.'
        # TODO: may check here if self._cycle is of type njet.extras.cderive
        cc = self._cycle.compose()
        
        max_power = kwargs.pop('max_power', min([e.hamiltonian.max_power for e in self.elements]))
        self._optics_taylor_map = taylor_map(*cc, max_power=max_power)
        
        _ = kwargs.setdefault('order', self._tpsa.order) # The order of the normal form procedure should be determined by the order of the Taylor map by default. A warning will be issued in 'fnf' if the requested order > self._tpsa.order
        kwargs.update(self._tpsa_input)
        return self._normalform(self._optics_taylor_map, **kwargs)
    