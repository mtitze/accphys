import numpy as np
import warnings

from njet.functions import cos
from lieops import create_coords, construct, poly, lexp
from lieops.solver.splitting import get_scheme_ordering
from lieops.solver import heyoka

from .tools import energy2beta0


# N.B. the length of an element will be used only later, when it comes to calculating the flow.

# Reference(s):
# [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
# [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)

class hard_edge_element:
    def __init__(self, *args, length=1, warn=True, **kwargs):
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
        self.length = length
        # Check user input regarding energy
        if 'energy' in kwargs.keys():
            _ = kwargs.setdefault('beta0', energy2beta0(kwargs['energy']))
        # Set max_power default to be provided to (possible) Lie-operator arguments (see lieops.core.poly)
        _ = kwargs.setdefault('max_power', 10)
            
        # Now calculate & set the Hamiltonian
        if 'beta0' in kwargs.keys():
            self.calcHamiltonian(*args, **kwargs)
        elif 'full_hamiltonian' in kwargs.keys(): # full_hamiltonian preference over Hamiltonian from beta0-calculation.
            self.full_hamiltonian = kwargs['full_hamiltonian']
        elif len(args) == 1: # In this case we assume theat args[0] represents a Hamiltonian or Lie-operator.
            a = args[0]
            if isinstance(a, poly):
                self.full_hamiltonian = a
            elif isinstance(a, lexp):
                self.full_hamiltonian = -a.argument/length
            else:
                raise RuntimeError(f'Argument {a} not recognized.')

        if hasattr(self, 'full_hamiltonian'):
            self.setProjection(**kwargs)
        else:
            if warn:
                warnings.warn("Hamiltonian not set. Provide 'beta0' or 'energy' argument.")
                    
    def calcHamiltonian(self, beta0, sqrtexp: int=2, tol=5e-8, **kwargs):
        '''
        Compute the Hamiltonian of a drift.
        For the underlying coordinate system and further details see Refs. [1, 2] below.
        In particular, the Hamiltonian is given by Eq. (2.16) in Ref. [2].

        Parameters
        ----------
        sqrtexp or power: int, optional
            Power up to which the square root of the drift should be expanded.
            
        tol: float, optional
            Threshold below which terms in the Hamiltonian are considered to be zero.
        '''
        assert 0 < beta0 and beta0 < 1
        # Compute the CFM drift part
        x, y, sigma, px, py, psigma = create_coords(3, real=True, **kwargs)
        one_hateta2 = lambda ps: ((1 + ps*beta0**2)**2 - 1 + beta0**2)/beta0**2 # Eqs. (15c) and (17) in Ref. [1]
        sqrt = lambda p1, p2, ps: (one_hateta2(ps) - p1**2 - p2**2)**(1/2)
        _ = kwargs.pop('power', None) # n.b. this will not remove the key in any calling instance
        drift_s = construct(sqrt, px, py, psigma, power=sqrtexp, **kwargs).above(tol) # expand sqrt around [px, py, psigma] = [0, 0, 0] up to power.
        hamiltonian = psigma - drift_s
        hamiltonian = hamiltonian.pop((0, 0, 0, 0, 0, 0), None).above(tol)
        self.full_hamiltonian = hamiltonian
        self._prop = {}
        self._prop['beta0'] = beta0
        self._prop['sqrtexp'] = sqrtexp
        self._prop['dE_E'] = psigma*beta0**2
        self._prop['drift'] = hamiltonian
        self._prop['drift_sqrt'] = drift_s
        self._prop['full'] = hamiltonian
        self._prop['coords'] = x, y, sigma, px, py, psigma
        
    def setProjection(self, *projection, tol=0, **kwargs):
        '''
        Set Hamiltonian by dropping components associated with terms not in given list.
        
        self.hamiltonian will be updated.
        
        !! Attention !!
        After this operation it may become necessary to re-caclulate flows and the one-turn map again.
        
        Parameters
        ----------
        projection: int
            The indices of the coordinates of interest. Must reach from 0 to self.full_hamiltonian.dim.
            If there are no arguments given, then the default Hamiltonian will be restored.
            
        tol: float, optional
            Threshold below which terms in the Hamiltonian are considered to be zero.
        '''
        # consistency checks
        self._projection = projection
        new_dim = len(projection)
        ham = self.full_hamiltonian
        if new_dim == 0: # default: 6D Hamiltonian
            projection = range(ham.dim)
            new_dim = ham.dim
        assert new_dim <= ham.dim, 'Number of requested dimension too large.'
        assert max(projection) < ham.dim, 'Requested dimension-index larger than any available index.'
        projection = list(projection) + [p + ham.dim for p in projection] # the eta-components duplicate the indices.
        complement = [k for k in range(2*ham.dim) if not k in projection]
        new_values = {}
        for k, v in ham.items():
            if any([k[p] != 0 for p in complement]): # only keep those coefficients which do not couple to other directions
                continue
            new_values[tuple([k[p] for p in projection])] = v
        if tol > 0:
            ham = ham.above(tol)
        self.hamiltonian = ham.__class__(values=new_values, dim=new_dim, max_power=ham.max_power)
        self.setOperator()
        
    def setOperator(self, **kwargs):
        '''
        Set the respective Lie-operator representing the current hard-edge model.
        '''
        self.operator = lexp(-self.hamiltonian*self.length)
        
    def apply(self, *args, **kwargs):
        '''
        Apply an operator on the current Hamiltonian; re-set the operator accordingly.
        '''
        self.hamiltonian = self.hamiltonian.apply(*args, **kwargs)
        self.setOperator() # re-set the operator
        
    def copy(self):
        result = self.__class__(length=self.length, warn=False)
        # copy all the fields
        for field, value in self.__dict__.items():
            if hasattr(value, 'copy'):
                setattr(result, field, value.copy())
            else:
                setattr(result, field, value)
        return result
    
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
            n_slices = int(np.ceil(self.length/kwargs['step']))
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
        for e in new_elements:
            e.length = self.length/n_slices
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

        new_elements = []
        for k in range(len(new_scheme)):
            f = new_scheme[k]
            if k%2 == 0:
                ham = ham1
            else:
                ham = ham2
            # As stated above, the selection of keys may break properties of derived classes of the hard_edge_element class, we create new hard_edge_elements instead.
            new_elements.append(hard_edge_element(ham, length=self.length/n_slices*f))

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
        
        if maintain_length:
            # We modify the Hamiltonians by using the total number of used parts, in order 
            # to maintain the overal length of the element. In principle one can also use negative 
            # lengths etc. coming from the splitting routine. But this would require that the 
            # user-given splitting routines provide these lengths somehow in their output formats.
            # We felt that this leads to more clumsy code and checks, so we opted for this solution here. 
            new_elements = [hard_edge_element(h*len(hamiltonians), length=self.length/len(hamiltonians)/n_slices) for h in hamiltonians]
        else:
            new_elements = [hard_edge_element(h, length=self.length/n_slices) for h in hamiltonians]
            
        return new_elements, split_order
    
    def __call__(self, *args, **kwargs):
        if not hasattr(self, 'operator'):
            self.setOperator(**kwargs)
        return self.operator(*args, **kwargs)
        
        
class phaserot(hard_edge_element):
    def __init__(self, *tunes, length=1, **kwargs):
        '''
        A generic uncoupled phase (space) rotation.
        
        Parameters
        ----------
        tunes: float or array_like
            Tune(s) defining the phase rotation.
        '''
        self.tunes = tunes
        self.length = length
        if len(tunes) > 0:
            self.calcHamiltonian(*tunes, **kwargs)
        
    def calcHamiltonian(self, *tunes, **kwargs):
        dim = len(tunes)
        xieta = create_coords(dim=dim)
        self.full_hamiltonian = sum([-tunes[k]*xieta[k]*xieta[k + dim] for k in range(dim)])
        
class cfm(hard_edge_element):
    def __init__(self, components=[0], tilt=0, _addzero=True, *args, **kwargs):
        '''
        Class to model a combined-function-magnetc (cfm).

        Parameters
        ----------
        beta0: float
            Relativistic beta = v/c of the reference trajectory.

        components: list, optional
            A list of complex entries, denoting the components of the cfm field.
            The components are hereby given with respect to a transversal axis, tilted against the Serre-Frenet
            coordinate system by an angle of choice (see tilt parameter below and Eq. (2.40) in Ref. [2]).

            If K_x = 0 this transverse axis smears out a cone. The ideal trajectory is hereby given by a horizontal
            cut of the cone with a plane.
            
            Attention:
            1) 
            Additional zeros in the list will cause the routine to compute more orders of the CFM.
            This becomes important if the cfm contains a non-zero dipole component and some other multipole components.
            
            2)
            In the case of a single entry in the cfm components, an additional zero will be added internally so that
            the resulting object describes a physical dipole. The reason is that (internally) the order of the G-function
            in Ref. [2] has to be at least 2 or higher (see the topmost Eq. on p.53 in Ref. [2]). To suppress this behaviour,
            the switch _addzero can be set to False.
            
        _addzero: boolean, optional
            Suppress the addition of a zero to the list of components in the case that len(components) = 1.
            
        tilt: float, optional
            The tilt between the dipole component and the higher-order magnet components of the cfm.
            
        **kwargs
            Optional arguments passed to self.calcHamiltonian and self.setProjection.
            
        Reference(s)
        ------------
            [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
            [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)
        '''
        assert len(components) > 0
        if len(components) == 1 and _addzero:
            components = [components[0], 0] # see explanation in the docs above
        self.components = components
        self.tilt = tilt
        hard_edge_element.__init__(self, *args, **kwargs)
        
    def setProjection(self, *args, **kwargs):
        '''
        Set self.hamiltonian to requested dimension.
        
        Parameters
        ----------
        style: str, optional
            Name of the key in self._prop denoting the Hamiltonian to be used.
            Supported options are:
            'full': use the full Hamiltonian (default)
            'kick': Only use the Hamiltonian containing field-components (see Eq. (2.36) in Ref. [2])
            'drift': Only the drift part of the Hamiltonian is used, i.e. all fields switched off.
        '''
        supported_styles = ['full', 'kick', 'drift']
        if not hasattr(self, '_style') or 'style' in kwargs.keys():
            self._style = kwargs.get('style', 'full')
        if not self._style in supported_styles:
            raise RuntimeError(f"Style '{self._style}' not recognized. Supported styles:\n{supported_styles}")
        self.full_hamiltonian = self._prop[self._style]
        hard_edge_element.setProjection(self, *args, **kwargs)
    
    def _g(self):
        '''
        Compute the g-part of the cfm. Code taken in parts from my MAD-X implementation 
        (see https://github.com/mtitze/MAD-X/blob/master/src/trrun.f90).
        '''
        exp = lambda z: complex(np.exp(z)) # convert numpy.float64 back to float as precaution to prevent unpredictable results if using jets etc.

        # Eq. (2.40) resp. (2.47) in Ref. [2]:
        # (By + 1j*Bx)_(phi = phi0) = sum_k components[k] r^k,
        # where phi0 = tilt here.

        # since kx = real(components[0]) and ky = -imag(components[0]), i.e.
        # components[0] = kx - 1j*ky,
        # we have:
        kappa = self.components[0].conjugate()
        barkappa = self.components[0]

        nord = len(self.components) - 1
        # Now fill up the g_{ij}'s for j = 0, ..., i and i = 0, ..., nord + 1.
        g = {}
        g[(0, 0)] = 0
        g[(1, 0)] = -barkappa
        g[(1, 1)] = -kappa # = g[(1, 0)].conjugate()
        for k in range(1, nord + 1):
            for j in range(k):
                # Eq. (6), in Ref. [1]
                g[(k + 1, j + 1)] = ( barkappa*g[(k, j + 1)]*(j + 1)*(j - k + 3/2) + 
                                     kappa*g[(k, j)]*(k - j)*(1/2 - j) )/(k - j)/(j + 1)
            # Eq. (8) in Ref. [1]
            sum0 = 0
            for j in range(1, k + 1):
                sum0 = sum0 - (k + 1 - j)*g[(k + 1, j)]*exp(-self.tilt*2*1j*j)
            g[(k + 1, 0)] = ( sum0 - 2**k*exp(-self.tilt*1j*k)*( self.components[k] 
                            + 1/2*(barkappa*exp(self.tilt*1j) + kappa*exp(self.tilt*-1j))*self.components[k - 1] ) )/(k + 1)
            g[(k + 1, k + 1)] = g[(k + 1, 0)].conjugate()
        return g
    
    def calcHamiltonian(self, tol=5e-8, **kwargs):
        '''
        Compute the Hamiltonian of the cfm (without electric fields).
        For the underlying coordinate system and further details see Refs. [1, 2] below.
        In particular, the Hamiltonian is given by Eq. (2.16) in Ref. [2].

        Parameters
        ----------
        tol: float, optional
            Threshold below which terms in the Hamiltonian are considered to be zero.
        
        **kwargs
            Optional keyword arguments passed to 'construct' routine.

        Returns
        -------
        dict
            A dictionary having the following keys:
            
            kx: float
                The field strength of the x-component of the dipole.
                
            ky: float
                The field strength of the y-component of the dipole.

            H: poly
                The Hamiltonian of the cfm up to the requested order.

            H_drift: poly
                The drift part of H.

            H_field: poly
                The field part of H.
                
            g: dict
                The g-components in Ref. [1]
            G: dict
                The G-components in Ref. [1]

            So that H = H_drift + H_field.
        '''
        # compute the Hamiltonian of the drift
        hard_edge_element.calcHamiltonian(self, tol=tol, **kwargs)
        x, y, sigma, px, py, psigma = self._prop['coords']
        # Compute the CFM vector potential
        # G = (1 + Kx*x + Ky*y)*A_t near Eq. (2).
        # Here G := G*e/p0 (The factor e/p0 can be absorbed in the coefficients later; see also the comment on page 4, right column)
        g = self._g()
        rp = (x + y*1j)/2
        rm = rp.conjugate()
        G = sum([rp**(k - j)*rm**j*g[(k, j)] for k, j in g.keys()]) # N.B.: We need to multiply g[(k, j)] from right in case their entries are jetpoly objects etc. They need to be added as the coefficients of the Lie polynomials. 
        
        # Also compute the derivatives of G
        drp_G = sum([(k - j)*rp**(k - j - 1)*rm**j*g[(k, j)] for k, j in g.keys() if k != j]) # the partial derivative of G with respect to r_+
        drm_G = sum([j*rp**(k - j)*rm**(j - 1)*g[(k, j)] for k, j in g.keys() if j != 0])
        dx_G = (drp_G + drm_G)/2
        dy_G = (-drp_G + drm_G)/2/1j
        
        # Assemble output Hamiltonians
        out = {}
        lamb0 = self.components[0]
        kx = (lamb0 + lamb0.conjugate())/2 # lamb0.real; .conjugate() works with floats and other objects better than .real etc.
        ky = (lamb0.conjugate() - lamb0)/2/1j # -lamb0.imag
        # N.B.: Hfull = psigma - (1 + kx*x + ky*y)*drift - G
        drift_s = self._prop['drift_sqrt'] # the square root expression in the Hamiltonian belonging to the drift
        H_drift = psigma - drift_s
        H_field = - drift_s*(x*kx + y*ky) - G
        H_full = H_drift + H_field
        
        # remove any remaining constants; they do not affect the equations of motion. Also drop all terms below the given threshold.
        H_full = H_full.pop((0, 0, 0, 0, 0, 0), None).above(tol)
        H_drift = H_drift.pop((0, 0, 0, 0, 0, 0), None).above(tol)
        H_field = H_field.pop((0, 0, 0, 0, 0, 0), None).above(tol)

        out = {}
        out['kx'] = kx
        out['ky'] = ky
        out['full'] = H_full
        out['kick'] = H_field
        out['G'] = G
        out['drp_G'] = drp_G
        out['drm_G'] = drm_G
        out['dx_G'] = dx_G
        out['dy_G'] = dy_G
        out['g'] = g
        self._prop.update(out)
        self.full_hamiltonian = H_full
        
    def _map(self, x, y, sigma, px, py, psigma, ds):
        r'''
        An implementation of map (2.33) in Ref. [2].
        
        Parameters
        ----------
            ds: \delta_s according to Eq. (2.33) in Ref. [2].
        '''
        dx_G_map, dy_G_map = self._prop['dx_G'], self._prop['dy_G']
        
        sqrt2 = float(np.sqrt(2))
        xi1 = (x + px*1j)/sqrt2
        xi2 = (y + py*1j)/sqrt2
        xi3 = (sigma + psigma*1j)/sqrt2
        eta1 = (x - px*1j)/sqrt2
        eta2 = (y - py*1j)/sqrt2
        eta3 = (sigma - psigma*1j)/sqrt2
        
        dx_G = dx_G_map(xi1, xi2, xi3, eta1, eta2, eta3)
        dy_G = dy_G_map(xi1, xi2, xi3, eta1, eta2, eta3)
        
        ux = px + dx_G*ds
        uy = py + dy_G*ds
        
        # Preparation steps
        beta0 = self._prop['beta0']
        dE_E = psigma*beta0**2 # = eta in Ref. [2].
        one_hateta2 = ((1 + dE_E)**2 - 1 + beta0**2)/beta0**2 # (1 + \hat \eta)**2; Eq. (2.17) in Ref. [2].
        kx, ky = self._prop['kx'], self._prop['ky']
        denominator = 1 + (kx**2 + ky**2)*ds**2
        XI = (ux*kx + uy*ky)/denominator*2*ds
        ZETA = (ux**2 + uy**2 - one_hateta2)/denominator
        h = -XI/2 + np.sqrt(XI**2/4 - ZETA)
        curv = 1 + kx*x + ky*y
        
        # Map (2.33) in Ref. [2]:
        xf = x + ds*curv*(ux/h + ds*kx)
        yf = y + ds*curv*(uy/h + ds*ky)
        sigmaf = sigma + ds - ds*curv*(1 + dE_E)/h
        pxf = ux + ds*kx*h
        pyf = uy + ds*ky*h
        psigmaf = psigma
        
        return xf, yf, sigmaf, pxf, pyf, psigmaf

class multipole(cfm):
    def __init__(self, fs=0, n: int=0, *args, **kwargs):
        '''
        Model of a multipole with exactly 2*n pole faces.
        
        Parameters
        ----------
        fs: float or complex, optional
            The field strength of the multipole. The real and imaginary part of 'fs' are understood to define the normal and skew components.
            These components are given with respect to the x-axis (tilt 0) and an axis which is rotated relative to x by an angle of pi/2/n.
        
        n: int, optional
            Defines the number 2*n of pole faces.
        '''
        if n == 0: # drift case
            components = [0]
        else: # n > 0
            components = [0]*max([2, n]) # max([2, n]) because if n = 1 (the dipole case) then, since (1 + Kx*x + Ky*y)*(By + 1j*Bx) = - \partial_x G + 1j*\partial_y G (see p. 53 top in Ref. [2]), G must be of order <= 2.
            #
            # Equivalently, these components are given by the real and imaginary parts of c_n with B_y + i B_x = c_n*(x + iy)**(n - 1), see e.g.
            # Eq. (1.8) in Ref. [2], using y=0 here. The c_n are just the components of the cfm.
            # Therefore:
            components[n - 1] = fs
        cfm.__init__(self, components=components, *args, **kwargs)
        
class drift(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, *args, n=0, **kwargs)
        
class dipole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, *args, n=1, **kwargs)
        
class quadrupole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, *args, n=2, **kwargs)
        
class sextupole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, *args, n=3, **kwargs)
    
class octupole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, *args, n=4, **kwargs)
        
class rfc(hard_edge_element):
    def __init__(self, voltage, phase, frequency, beta0, *args, **kwargs):
        '''
        A generic RF cavity.
                
        Reference(s):
        [1] A. Wolski: Beam Dynamics in High Energy Particle Accelerators.
        '''
        assert 0 < beta0 and beta0 < 1
        self.voltage = voltage
        self.phase = phase
        self.frequency = frequency # Actually, translated to: k = 2*np.pi/constants.speed_of_light*1/T
        self.beta0 = beta0
        hard_edge_element.__init__(self, *args, beta0=beta0, **kwargs)
        
    def calcHamiltonian(self, p=10, tol=5e-8, **kwargs):
        '''
        The Hamiltonian of a simplified RF cavity., see Ref. [1], Eq. (3.138) p. 112.
        
        Parameters
        ----------
        p: int, optional
            The total order up to which the RF potential should be expanded in terms of polynomials.
        
        tol: float, optional
            Threshold below which terms in the Hamiltonian are considered to be zero.
        '''
        hard_edge_element.calcHamiltonian(self, tol=tol, **kwargs)
        x, y, sigma, px, py, psigma = self._prop['coords']
        #k = 2*np.pi*self.frequency/constants.speed_of_light # 2.40483/radius # Eq. (3.132) in [1] and earlier: omega = k/c
        #hamiltonian = construct(cos, sigma/beta0*-k + self.phase, **kwargs)*self.voltage
        rf_potential = construct(cos, -sigma/self.beta0*self.frequency + self.phase, power=p, **kwargs)*self.voltage/float(np.pi)
        hamiltonian = self._prop['full'] - rf_potential
        self.full_hamiltonian = hamiltonian.pop((0, 0, 0, 0, 0, 0), None).above(tol) # remove any constant term
        
        
        
        
    

        