import numpy as np

from njet.functions import cos
from lieops import create_coords, construct

# N.B. the length of an element will be used only later, when it comes to calculating the flow.

# Reference(s):
# [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
# [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)

class hard_edge_element:
    
    def __init__(self, *args, length=1, **kwargs):
        '''
        Class to model the Hamiltonian of a 6D hard-edge element.
        '''
        self.length = length
        self.calcHamiltonian(*args, **kwargs)
        self.setHamiltonian(**kwargs)
        
    def setHamiltonian(self, *projection, **kwargs):
        '''
        Set Hamiltonian by dropping components associated with terms not in given list.
        
        self.hamiltonian will be updated.
        
        Parameters
        ----------
        projection: int
            The indices of the coordinates of interest. Must reach from 0 to self.full_hamiltonian.dim.
            If there are no arguments given, then the default Hamiltonian will be restored.
        '''
        # consistency checks
        new_dim = len(projection)
        ham = self.full_hamiltonian
        if new_dim == 0: # default: 6D Hamiltonian
            projection = range(ham.dim)
            new_dim = ham.dim
        assert new_dim <= ham.dim and max(projection) < ham.dim, 'Number of spatial dimensions too large.'
        projection = list(projection) + [p + ham.dim for p in projection] # the eta-components duplicate the indices.
        complement = [k for k in range(2*ham.dim) if not k in projection]
        new_values = {}
        for k, v in ham.items():
            if any([k[p] != 0 for p in complement]): # only keep those coefficients which do not couple to other directions
                continue
            new_values[tuple([k[p] for p in projection])] = v
        self.hamiltonian = ham.__class__(values=new_values, dim=new_dim, max_power=ham.max_power)
        
    def calcHamiltonian(self, beta0, sqrtexp: int=2, **kwargs):
        '''
        Compute the Hamiltonian of a drift.
        For the underlying coordinate system and further details see Refs. [1, 2] below.
        In particular, the Hamiltonian is given by Eq. (2.16) in Ref. [2].

        Parameters
        ----------
        sqrtexp or power: int, optional
            Power up to which the square root of the drift should be expanded. 
        '''
        kwargs['power'] = kwargs.get('power', sqrtexp)
        # Compute the CFM drift part
        x, y, sigma, px, py, psigma = create_coords(3, real=True, **kwargs)
        one_hateta2 = lambda ps: ((1 + ps*beta0**2)**2 - 1 + beta0**2)/beta0**2 # Eqs. (15c) and (17) in Ref. [1]
        sqrt = lambda p1, p2, ps: (one_hateta2(ps) - p1**2 - p2**2)**(1/2)
        drift_s = construct(sqrt, px, py, psigma, **kwargs) # expand sqrt around [px, py, psigma] = [0, 0, 0] up to power
        hamiltonian = psigma - drift_s
        hamiltonian.pop((0, 0, 0, 0, 0, 0), None)
        self.full_hamiltonian = hamiltonian
        self._prop = {}
        self._prop['drift'] = hamiltonian
        self._prop['drift_sqrt'] = drift_s
        self._prop['full'] = hamiltonian
        self._prop['coords'] = x, y, sigma, px, py, psigma
        
        
class phaserot(hard_edge_element):
    def __init__(self, *tunes, **kwargs):
        '''
        A generic uncoupled phase rotation.
        
        Parameters
        ----------
        tunes: float
            Tune(s) defining the phase rotation.
        '''
        self.tunes = tunes
        hard_edge_element.__init__(self, *tunes, **kwargs)
        
    def calcHamiltonian(self, *tunes, **kwargs):
        dim = len(tunes)
        xieta = create_coords(dim=dim)
        self.full_hamiltonian = sum([-tunes[k]*xieta[k]*xieta[k + dim] for k in range(dim)])

        
class cfm(hard_edge_element):
    def __init__(self, beta0, components=[0], tilt=0, *args, **kwargs):
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
            
            Note that additional zeros in the list will cause the routine to compute more orders of the CFM.
            
        tilt: float, optional
            The tilt between the dipole component and the higher-order magnet components of the cfm.
            
        **kwargs
            Optional arguments passed to self.calcHamiltonian and self.setHamiltonian.
        '''
        assert 0 < beta0 and beta0 < 1
        self.components = components
        self.tilt = tilt
        self.beta0 = beta0
        hard_edge_element.__init__(self, *args, beta0=beta0, **kwargs)
        
    def setHamiltonian(self, *args, **kwargs):
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
        if not hasattr(self, '_style') or 'style' in kwargs.keys():
            self._style = kwargs.get('style', 'full')
        self.full_hamiltonian = self._prop[self._style]
        hard_edge_element.setHamiltonian(self, *args, **kwargs)
    
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

    def calcHamiltonian(self, **kwargs):
        '''
        Compute the Hamiltonian of the cfm (without electric fields).
        For the underlying coordinate system and further details see Refs. [1, 2] below.
        In particular, the Hamiltonian is given by Eq. (2.16) in Ref. [2].

        Parameters
        ----------
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

            H: liepoly
                The Hamiltonian of the cfm up to the requested order.

            H_drift: liepoly
                The drift part of H.

            H_field: liepoly
                The field part of H.
                
            g: dict
                The g-components in Ref. [1]
            G: dict
                The G-components in Ref. [1]

            So that H = H_drift + H_field.
        '''
        # compute the Hamiltonian of the drift
        hard_edge_element.calcHamiltonian(self, **kwargs)
        x, y, sigma, px, py, psigma = self._prop['coords']
        # Compute the CFM vector potential
        # G = (1 + Kx*x + Ky*y)*A_t near Eq. (2).
        # Here G := G*e/p0 (The factor e/p0 can be absorbed in the coefficients later; see also the comment on page 4, right column)
        g = self._g()
        rp = (x + y*1j)/2
        rm = rp.conjugate()
        G = sum([rp**(k - j)*rm**j*g[(k, j)] for k, j in g.keys()]) # N.B.: We need to multiply g[(k, j)] from right in case their entries are jetpoly objects etc. They need to be added as the coefficients of the Lie polynomials. 
        
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
        
        # remove any remaining constants; they do not affect the equations of motion.
        H_full.pop((0, 0, 0, 0, 0, 0), None)
        H_drift.pop((0, 0, 0, 0, 0, 0), None)
        H_field.pop((0, 0, 0, 0, 0, 0), None)

        out = {}
        out['kx'] = kx
        out['ky'] = ky
        out['full'] = H_full
        out['kick'] = H_field
        out['G'] = G
        out['g'] = g
        self._prop.update(out)
        self.full_hamiltonian = H_full

class multipole(cfm):
    def __init__(self, str=0, n: int=0, *args, **kwargs):
        '''
        Model of a multipole with exactly 2*n pole faces.
        
        Parameters
        ----------
        n: int, optional
            Defines the number 2*n of pole faces.

         
        expansion: int, optional
            An additional parameter (>= n) by which one can control the number of terms in the expansion of the field.
        '''
        if n == 0: # drift case
            components = [0]
        else: # n > 0
            components = [0]*max([2, n]) # max([2, n]) because if n = 1 (the dipole case) then, since (1 + Kx*x + Ky*y)*(By + 1j*Bx) = - \partial_x G + 1j*\partial_y G (see p. 53 top in Ref. [2]), G must be of order <= 2.
            # The real and imaginary part of 'str' are understood to define the normal and skew components of the sextupole.
            # These components are given with respect to the x-axis (tilt 0) and an axis which is rotated relative to x by an angle of pi/2/n.
            #
            # Equivalently, these components are given by the real and imaginary parts of c_n with B_y + i B_x = c_n*(x + iy)**(n - 1), see e.g.
            # Eq. (1.8) in Ref. [2], using y=0 here. The c_n are just the components of the cfm.
            # Therefore:
            components[n - 1] = str
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
        A generic RF cavity based on the simplest of all models.
                
        Reference(s):
        [1] A. Wolski: Beam Dynamics in High Energy Particle Accelerators.
        '''
        assert 0 < beta0 and beta0 < 1
        self.voltage = voltage
        self.phase = phase
        self.frequency = frequency # Actually, translated to: k = 2*np.pi/constants.speed_of_light*1/T
        self.beta0 = beta0
        hard_edge_element.__init__(self, *args, beta0=beta0, **kwargs)
        
    def calcHamiltonian(self, p=10, **kwargs):
        '''
        The Hamiltonian of a simplified RF cavity., see Ref. [1], Eq. (3.138) p. 112.
        '''
        hard_edge_element.calcHamiltonian(self, **kwargs)
        x, y, sigma, px, py, psigma = self._prop['coords']
        #k = 2*np.pi*self.frequency/constants.speed_of_light # 2.40483/radius # Eq. (3.132) in [1] and earlier: omega = k/c
        #hamiltonian = construct(cos, sigma/beta0*-k + self.phase, **kwargs)*self.voltage
        rf_potential = construct(cos, -sigma/self.beta0*self.frequency + self.phase, power=p, **kwargs)*self.voltage/float(np.pi)
        hamiltonian = self._prop['full'] - rf_potential
        hamiltonian.pop((0, 0, 0, 0, 0, 0), None) # remove any constant term
        self.full_hamiltonian = hamiltonian
        
        
        
    

        