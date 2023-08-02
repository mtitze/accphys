import warnings
import numpy as np
from copy import deepcopy

from lieops import lexp, poly
from lieops.solver.splitting import get_scheme_ordering

class hard_edge_element:
    def __init__(self, *args, warn=True, **kwargs):
        '''
        Class to model the Hamiltonian of a 6D hard-edge element.
        
        Parameters
        ----------
        *args
            If also beta0 is given, then these will be the parameters of self.calcHamiltonian. 
            For example the field strengths of the magnets (check out the respective classes in elements.py).
            
        length: float, optional
            The length of the element.
            
        energy: float, optional
            Energy in GeV required to calculate the Hamiltonians.
            Attention: 
            By default, the electron mass is used. This can
            be changed in accphys.tools.energy2beta0. Alternatively,
            the beta0 keyword can be used for other type of particles.

        beta0: float, optional
            Relativistic beta required to calculate the Hamiltonians.
            Will get priority over 'energy' keyword, if provided.
            
        warn: boolean, optional
            Enable/disable warnings when initiating this object.
            
        **kwargs
            Optional parameters passed to the calculation of the Hamiltonian.
        '''
        assert len(args) <= 1, 'More than one non-keyworded arguments provided.'
        # Set user-given data
        for k, v in kwargs.items():
            setattr(self, str(k), v)

        # User may provide the Hamiltonian or Lie-operator directly:
        if len(args) == 1 and 'hamiltonian' not in kwargs.keys() and not hasattr(self, 'hamiltonian'): 
            a = args[0]
            if isinstance(a, poly):
                self.hamiltonian = a
                self.setOperator(warn=warn)
            elif isinstance(a, lexp):
                self.operator = a
                length = getattr(self, 'length', 1)
                if length != 0:
                    self.hamiltonian = -a.argument/length
                else:
                    self.hamiltonian = -a.argument
            else:
                raise RuntimeError(f'Argument {a} not recognized.')

        if not hasattr(self, 'hamiltonian') and hasattr(self, 'calcHamiltonian'):
            self.calcHamiltonian(**kwargs)

        if not hasattr(self, 'hamiltonian') and warn:
            warnings.warn("Hamiltonian not set.")

        if hasattr(self, 'hamiltonian'): # even if self.operator exists, overwrite it
            self.setOperator(warn=warn)
                    
    def project(self, *projection, tol_drop=0, **kwargs):
        '''
        Project the current hamiltonian to a specific phase space subset, 
        by dropping components associated with terms not in given list.
        
        Parameters
        ----------
        projection: int
            The indices of the coordinates of interest. Must reach from 0 to self.full_hamiltonian.dim.
            If there are no arguments given, then the default Hamiltonian will be restored.
            
        tol_drop: float, optional
            Threshold below which terms in the Hamiltonian are considered to be zero.
        '''
        # consistency checks
        new_fields = {}
        for k, v in self.__dict__.items():
            if isinstance(v, poly):
                nv = v.project(*projection)
                if tol_drop > 0:
                    nv = nv.above(tol_drop)
                new_fields[k] = nv
            else:
                new_fields[k] = deepcopy(v)
        return self.__class__(warn=False, **new_fields)
        
    def setOperator(self, warn=True, psfactor=-1):
        '''
        Set the respective Lie-operator representing the current hard-edge model.
        
        If H denotes the Hamiltonian of the element, then it holds in (complex) xi/eta-coordinates:
           dz_j/dt = -1j {z_j, H} ,     
        where the Poisson-bracket is taken with respect to xi/eta. Usually the Hamiltonian H
        is already cast into its xi/eta-form: H0 := -1j*H. Then the above reads:
           dz_j/dt = {z_j, H0} .
        This system has the formal solution
           z_j^(final) = exp(-:H0:*length) z_j^(initial) .     
        This routine will construct a Lie-operator with the above argument.
        
        Parameters
        ----------
        warn: boolean, optional
            Show a warning in case of thin-lens elements.
            
        psfactor: float, optional
            The factor in front of the Hamiltonian to be multiplied with. By default this factor is "-1", as
            outlined above, but could in principle be also changed to "1j" or another value.
            In its default setting, is therefore important that the Hamiltonian is cast with respect
            to its complex xi/eta-coordinates.

        Atttention: 
        1) If a length is provided to the current model, this length will
           be taken into account when setting the Lie-operator:
           operator = lexp(psfactor*hamiltonian*length)
        2) If the element has length 0, then the length will be ignored and a warning will
           be issued.
           operator = lexp(psfactor*hamiltonian)
        '''
        length = getattr(self, 'length', 1)
        if length != 0:
            self.operator = lexp(self.hamiltonian*length*psfactor)
        else:
            # lengths of 'thin-lens' elements are ignored.
            if warn:
                warnings.warn('Lie-operator of thin element set to "lexp(-hamiltonian)".')  
            self.operator = lexp(self.hamiltonian*psfactor)
        
    def apply(self, *args, **kwargs):
        '''
        Apply an operator on the current Hamiltonian; re-set the operator accordingly.
        '''
        self.hamiltonian = self.hamiltonian.apply(*args, **kwargs)
        self.setOperator() # re-set the operator
        
    def copy(self):
        inp = {}
        # copy all the fields, if a copy functionaliy exists
        for field, value in self.__dict__.items():
            if hasattr(value, 'copy'):
                inp[field] = value.copy()
            else:
                inp[field] = value
        return self.__class__(warn=False, **inp)
    
    def split(self, n_slices: int=1, return_scheme_ordering=False, **kwargs):
        '''
        Split the element into several elements of various lengths.
                
        Parameters
        ----------
        scheme: list, optional
            A list of coefficients [a1, b1, a2, b2, ...] so that an individual slice of length h
            of the given Hamiltonian H = H1 + H2 has the form:
            exp(h*H) = exp(h*a1*H1) o exp(h*b1*H2) o exp(h*a2*H1) o exp(h*b2*H2) o ...
            
        n_slices: int, optional
            The number of requested slices, so that we have
            exp(H) = exp(h*H)**n_slices
            
        step: float, optional
            If given, split the individual elements according to a given length. This
            is intended to reduce the number of slices for short elements and
            find a proper number of slices for long elements. If given, the n_slices parameter
            will be ignored.

        combine: boolean, optional
            Sum adjacent operators of the same type (within the same element), if possible.
            Note that operators belonging to different elements will not be combined (as they do not
            commute in general).
            
        method: callable, optional
            A custom method to split a given Hamiltonian into several parts. It should return the new
            Hamiltonians.
            
        Returns
        -------
        list
            A list of hard_edge_element objects, representing a slicing of the current element.
        '''
        # overwrite n_slices if user provides a 'step' parameter
        if 'step' in kwargs.keys():
            length = getattr(self, 'length', 1)
            n_slices = max([1, int(np.ceil(length/kwargs['step']))])
            _ = kwargs.pop('step')
        assert n_slices >= 1
        
        if 'method' in kwargs.keys():
            # split the element into several elements according to the provided method.
            elements, scheme_ordering = self._split_into_custom_elements(n_slices=n_slices, **kwargs)
        elif 'scheme' in kwargs.keys() and 'keys' in kwargs.keys():
            # split the element alternatingly into two kinds of elements according to the given scheme and the requested keys.
            elements, scheme_ordering = self._split_into_alternating_elements(n_slices=n_slices, **kwargs)
        else:
            # split the element uniformly into slices.
            elements, scheme_ordering = self._split_into_slices(n_slices=n_slices, **kwargs)
            
        if return_scheme_ordering:
            return elements, scheme_ordering
        else:
            return elements
        
    def _split_into_slices(self, n_slices: int=1, **kwargs):
        new_elements = [self.copy()]*n_slices
        length = getattr(self, 'length', 1)
        for e in new_elements:
            e.length = length/n_slices
            e.setOperator()
        return new_elements, [0]*n_slices
    
    def _split_into_alternating_elements(self, n_slices: int=1, combine=True, **kwargs):
        # N.B. below we have to use the hard_edge_element class explicitly, because if we would have used
        # self.__class__, then derived classes will interpret the argument h differently. Besides, the result
        # of the splitting should not be considered as some derived subclass like a cfm.

        # We also do not use the poly.split routine at this point here, because we want to store all the new lengths in the respective elements and keep the individual hamiltonians unchanged. This may become useful at another step, where only the integration lengths are changed.
        
        if 'scheme' in kwargs.keys() and not 'keys' in kwargs.keys():
            warnings.warn("Splitting with 'scheme' parameter requires 'keys' parameter to be set.")
        
        scheme = kwargs.get('scheme', [0.5, 1, 0.5])
        keys = kwargs.get('keys', [])
        ### end of check
        
        ham1 = self.hamiltonian.extract(key_cond=lambda x: x in keys)
        ham2 = self.hamiltonian.extract(key_cond=lambda x: x not in keys)
        if ham1 == 0 or ham2 == 0:
            # in this case we just return a slicing of the original element
            return self._split_into_slices(**kwargs)

        scheme = list(scheme)
        if len(scheme)%2 == 1 and n_slices > 1 and combine:
            # In this case the given scheme of coefficients, belonging to terms of the two operators, 
            # have its end and start belonging to the same operator. For more than one slice they can therefore
            # be combined together, which is done here.
            start = [scheme[0]]
            center = scheme[1:]
            center[-1] *= 2
            end = scheme[1:]
            new_scheme = start + center*(n_slices - 1) + end
        else:
            # len(scheme) even or n_slices == 1
            new_scheme = scheme*n_slices

        length = getattr(self, 'length', 1)
        new_elements = []
        for k in range(len(new_scheme)):
            f = new_scheme[k]
            if k%2 == 0:
                ham = ham1
            else:
                ham = ham2
            # As stated above, the selection of keys may break properties of derived classes of the hard_edge_element class, we create new hard_edge_elements instead.
            new_elements.append(hard_edge_element(ham, length=length/n_slices*f))

        return new_elements, get_scheme_ordering(new_scheme)
    
    def _split_into_custom_elements(self, method, n_slices: int=1, maintain_length=False, **kwargs):
        '''
        Split an element into custom elements according to a user-defined splitting routine.
        
        Parameters
        ----------
        method: callable
            A function mapping a poly object to a list of poly objects, defining the splitting.
            
        n_slices: integer, optional
            An optional additional slicing of the element into n_slices. This may be useful because
            a splitting routine is just an approximation to the original thick element which gets better
            the more slices are used.
            
        maintain_length: boolean, optional
            If true, then maintain the length of the original element, by multiplying the splitted
            Hamiltonians by N, where N denotes the number of elements, and the individual lengths by 1/N,
            respectively.
            
        Returns
        -------
        list
            A list of the new elements in form of hard_edge_element objects.
            
        list
            A list denoting the ordering of these elements.
        '''
        hamiltonians = method(self.hamiltonian, **kwargs)
        # determine the new scheme and set the new elements
        assert len(hamiltonians) > 0
        split_order = []
        unique_hamiltonians = []
        order_number = 0
        for h in hamiltonians:
            if h not in unique_hamiltonians:
                unique_hamiltonians.append(h)
                split_order.append(order_number)
                order_number += 1
            else:
                split_order.append(unique_hamiltonians.index(h))
            
        split_order = split_order*n_slices

        length = getattr(self, 'length', 1)
        if maintain_length:
            # We modify the Hamiltonians by using the total number of used parts, in order 
            # to maintain the overal length of the element. In principle one can also use negative 
            # lengths etc. coming from the splitting routine. But this would require that the 
            # user-given splitting routines provide these lengths somehow in their output formats.
            # We felt that this leads to more clumsy code and checks, so we opted for this solution here.
            new_elements = [hard_edge_element(h*len(hamiltonians), length=length/len(hamiltonians)/n_slices) for h in hamiltonians]
        else:
            new_elements = [hard_edge_element(h, length=length/n_slices) for h in hamiltonians]
            
        return new_elements, split_order
    
    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'operator'):
            self.setOperator(**kwargs)
        return self.operator(*args, **kwargs)