import numpy as np

from lieops import create_coords, construct

# N.B. the length of an element will be used only later, when it comes to calculating the flow.

class phaserot:
    def __init__(self, *tunes, length=1):
        '''
        A generic uncoupled phase rotation.
        
        Parameters
        ----------
        tunes: float
            Tune(s) defining the phase rotation.
        '''
        self.tunes = tunes
        self.length = length
        
        dim = len(tunes)
        xieta = create_coords(dim=dim)
        self.hamiltonian = sum([-tunes[k]*xieta[k]*xieta[k + dim] for k in range(dim)])

class cfm:
    def __init__(self, beta0, components=[0], tilt=0, length=1, dim: int=6, **kwargs):
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
            
        tilt: float, optional
            The tilt between the dipole component and the higher-order magnet components of the cfm.
            
        **kwargs
            Optional arguments passed to self.calcHamiltonian and self.setHamiltonian.
            
        Reference(s):
        [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
        [2] M. Titze: "Space Charge Modeling at the Integer Resonances for the CERN PS and SPS", PhD Thesis (2019)
        '''
        assert 0 < beta0 and beta0 < 1
        self.components = components
        self.tilt = tilt
        self.beta0 = beta0
        self.length = length
        
        self._prop = self.calcHamiltonian(**kwargs)
        self.setHamiltonian(dim=dim, **kwargs)
        
    def setHamiltonian(self, dim: int=6, style='full', **kwargs):
        '''
        Set self.hamiltonian to requested dimension.
        
        Parameters
        ----------
        dim: int, optional
            Dimension of the Hamiltonian to be set. Supported: 6, 4, 2.
        
        style: str, optional
            Name of the key in self._prop denoting the Hamiltonian to be used.
            Supported options are:
            'full': use the full Hamiltonian (default)
            'kick': Only use the Hamiltonian containing field-components (see Eq. (2.36) in Ref. [2])
            'drift': Only the drift part of the Hamiltonian is used, i.e. all fields switched off.
        '''
        ham = self._prop[style]
        if dim == 6:
            self._setHamiltonian6d(ham)
        elif dim == 4:
            self._setHamiltonian4d(ham)
        elif dim == 2:
            self._setHamiltonian2d(ham)
        else:
            raise NotImplementedError(f'No rule for Hamiltonian of dim: {dim}')
        
    def _setHamiltonian6d(self, ham):
        '''
        Set 6D Hamiltonian.
        '''
        self.hamiltonian = ham
        
    def _setHamiltonian4d(self, ham):
        '''
        Set 4d Hamiltonian by dropping those terms which belong to the (sigma, psigma)-pair.
        '''
        new_values = {}
        for k, v in ham.items():
            if k[2] != 0 or k[5] != 0:
                continue
            new_key = (k[0], k[1], k[3], k[4])
            new_values[new_key] = v
        self.hamiltonian = ham.__class__(values=new_values, dim=2, max_power=ham.max_power)
        
    def _setHamiltonian2d(self, ham):
        '''
        Set 2d Hamiltonian by dropping those terms which belong to the (y, py) and (sigma, psigma)-pairs.
        '''
        new_values = {}
        for k, v in ham.items():
            if k[1] != 0 or k[4] != 0 or k[2] != 0 or k[5] != 0:
                continue
            new_key = (k[0], k[3])
            new_values[new_key] = v
        self.hamiltonian = ham.__class__(values=new_values, dim=1, max_power=ham.max_power)
    
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

    def calcHamiltonian(self, sqrtexp: int=2, **kwargs):
        '''
        Compute the Hamiltonian of the cfm (without electric fields).
        For the underlying coordinate system and further details see Refs. [1, 2] below.
        In particular, the Hamiltonian is given by Eq. (2.16) in Ref. [2].

        Parameters
        ----------
        sqrtexp or power: int, optional
            Power up to which the square root of the drift should be expanded.
            
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

        Reference(s):
        [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
        [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)
        '''
        kwargs['power'] = kwargs.get('power', sqrtexp)
        # Compute the CFM drift part
        x, y, sigma, px, py, psigma = create_coords(3, cartesian=True, **kwargs)
        one_hateta2 = lambda ps: ((1 + ps*self.beta0**2)**2 - 1 + self.beta0**2)/self.beta0**2 # Eqs. (15c) and (17) in Ref. [1]
        sqrt = lambda p1, p2, ps: (one_hateta2(ps) - p1**2 - p2**2)**(1/2)
        drift_s = construct(sqrt, px, py, psigma, **kwargs) # expand sqrt around [px, py, psigma] = [0, 0, 0] up to power sqrtexp
        drift_s.pop((0, 0, 0, 0, 0, 0), None) # remove the constant emerging from the expansion.

        # Compute the CFM vector potential
        # G = (1 + Kx*x + Ky*y)*A_t near Eq. (2).
        # Here G := G*e/p0 (The factor e/p0 can be absorbed in the coefficients later; see also the comment on page 4, right column)
        g = self._g()
        rp = (x + y*1j)/2
        rm = rp.conjugate()
        G = sum([rp**(k - j)*rm**j*g[(k, j)] for k, j in g.keys()]) # need to multiply g[(k, j)] from right in case their entries are jetpoly objects etc. They need to be added as the coefficients of the Lie polynomials. 

        # Assemble output Hamiltonians
        out = {}
        lamb0 = self.components[0]
        kx = (lamb0 + lamb0.conjugate())/2 # lamb0.real; .conjugate() works with floats and other objects better than .real etc.
        ky = (lamb0.conjugate() - lamb0)/2/1j # -lamb0.imag
        # N.B.: Hfull = psigma - (1 + kx*x + ky*y)*drift - G
        H_drift = psigma - drift_s
        H_field = - drift_s*(x*kx + y*ky) - G
        H_full = H_drift + H_field
        out = {}
        out['kx'] = kx
        out['ky'] = ky
        out['full'] = H_full
        out['drift'] = H_drift
        out['kick'] = H_field
        out['G'] = G
        out['g'] = g
        return out
        
class multipole(cfm):
    def __init__(self, fx=0, fy=0, n: int=0, *args, **kwargs):
        '''
        Model of a multipole with n poles.
        
        Parameters
        ----------
        fx: float, optional
            x-component of the field strength with respect to the transverse (y=0)-plane.
            
        fy: float, optional
            y-component of the field strength with respect to the transverse (y=0)-plane.
            
        n: int, optional
            Define the number of poles. The resulting magnet will have 2*n poles.
            
        '''
        if n == 0:
            components = [0]
        else: # n > 0
            components = [0]*n
            components[n - 1] = fx - fy*1j
        cfm.__init__(self, components=components, *args, **kwargs)
        
class drift(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, n=0, *args, **kwargs)
        
class dipole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, n=1, *args, **kwargs)
        
class quadrupole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, n=2, *args, **kwargs)
        
class sextupole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, n=3, *args, **kwargs)
    
class octupole(multipole):
    def __init__(self, *args, **kwargs):
        multipole.__init__(self, n=4, *args, **kwargs)

        