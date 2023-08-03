import numpy as np
import warnings

from .common import hard_edge_element
from .drift import DriftHamiltonian

def _g(components, tilt=0):
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
    kappa = components[0].conjugate()
    barkappa = components[0]

    # Now fill up the g_{ij}'s for j = 0, ..., i and i = 0, ..., len(components) - 1.
    g = {}
    g[(0, 0)] = 0
    g[(1, 0)] = -barkappa
    g[(1, 1)] = -kappa # = g[(1, 0)].conjugate()
    for k in range(1, len(components) + 1): # k reflects the order up to which we consider the G-function
        for j in range(k):
            # Eq. (6), in Ref. [1]
            g[(k + 1, j + 1)] = ( barkappa*g[(k, j + 1)]*(j + 1)*(j - k + 3/2) + 
                                 kappa*g[(k, j)]*(k - j)*(1/2 - j) )/(k - j)/(j + 1)
        # Eq. (8) in Ref. [1]
        sum0 = 0
        for j in range(1, k + 1):
            sum0 = sum0 - (k + 1 - j)*g[(k + 1, j)]*exp(-tilt*2*1j*j)

        g[(k + 1, 0)] = sum0/(k + 1)
        # Add additional contributions from components, if they exist.
        if k < len(components) + 1:
            real = (barkappa*exp(tilt*1j) + kappa*exp(tilt*-1j))/2 # = Re(barkappa*exp(tilt*1j))
            fk = -2**k*exp(tilt*-1j*k)
            g[(k + 1, 0)] += fk*real*components[k - 1]/(k + 1)
            if k < len(components):
                g[(k + 1, 0)] += fk*components[k]/(k + 1)

        g[(k + 1, k + 1)] = g[(k + 1, 0)].conjugate()
    return g

def CFMHamiltonian(components, tilt=0, tol_drop=0, **kwargs):
    '''
    Compute the Hamiltonian of the cfm (without electric fields).
    For the underlying coordinate system and further details see Refs. [1, 2] below.
    In particular, the Hamiltonian is given by Eq. (2.16) in Ref. [2].

    Parameters
    ----------
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
        
    tilt: float, optional
        The tilt between the dipole component and the higher-order magnet components of the cfm.
    
    tol_drop: float, optional
        Threshold below terms in the Hamiltonian will be dropped.
    
    **kwargs
        Optional keyword arguments passed to 'elements.drift.DriftHamiltonian'.

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
        
    References
    ----------
    [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
    [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)
    '''
    # Compute the Hamiltonian of the drift
    DH = DriftHamiltonian(tol_drop=tol_drop, **kwargs)
    x, y, sigma, px, py, psigma = DH['coords']
    # Compute the CFM vector potential
    # G = (1 + Kx*x + Ky*y)*A_t near Eq. (2).
    # Here G := G*e/p0 (The factor e/p0 can be absorbed in the coefficients later; see also the comment on page 4, right column)
    g = _g(components=components, tilt=tilt)
    rp = (x + y*1j)/2
    rm = rp.conjugate()
    G = sum([rp**(k - j)*rm**j*g[(k, j)] for k, j in g.keys()]) # N.B.: We need to multiply g[(k, j)] from right in case their entries are jetpoly objects etc. They need to be added as the coefficients of the Lie polynomials. 
    
    # Also compute the derivatives of G
    drp_G = sum([(k - j)*rp**(k - j - 1)*rm**j*g[(k, j)] for k, j in g.keys() if k != j]) # the partial derivative of G with respect to r_+
    drm_G = sum([j*rp**(k - j)*rm**(j - 1)*g[(k, j)] for k, j in g.keys() if j != 0])
    dx_G = (drp_G + drm_G)/2
    dy_G = (-drp_G + drm_G)/2/1j
    
    # Assemble output Hamiltonians
    lamb0 = components[0]
    kx = (lamb0 + lamb0.conjugate())/2 # lamb0.real; .conjugate() works with floats and other objects better than .real etc.
    ky = (lamb0.conjugate() - lamb0)/2/1j # -lamb0.imag
    # N.B.: Hfull = psigma - (1 + kx*x + ky*y)*drift - G
    drift_s = DH['drift_sqrt'] # the square root expression in the Hamiltonian belonging to the drift
    H_drift = psigma - drift_s
    H_field = - drift_s*(x*kx + y*ky) - G
    H_full = H_drift + H_field
    
    # remove any remaining constants; they do not affect the equations of motion. Also drop all terms below the given threshold.
    H_full = H_full.pop((0, 0, 0, 0, 0, 0), None).above(tol_drop)
    H_drift = H_drift.pop((0, 0, 0, 0, 0, 0), None).above(tol_drop)
    H_field = H_field.pop((0, 0, 0, 0, 0, 0), None).above(tol_drop)

    out = DH
    out['components'] = components
    out['tilt'] = tilt
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
    out['hamiltonian'] = H_full
    
    if kx != 0:
        out['rhox'] = 1/kx # kx = 1/r
    else:
        out['rhox'] = float(np.inf)
        
    if ky != 0:
        out['rhoy'] = 1/ky
    else:
        out['rhoy'] = float(np.inf)
    
    return out
    
def _map(CFMH, x, y, sigma, px, py, psigma, ds):
    r'''
    An implementation of map (2.33) in Ref. [2].
    
    Parameters
    ----------
    CFMH: dict
        The output of CFMHamiltonian, containing the Hamiltonian of the combined-function-magnet and additional information.
    
    ds: float
        \delta_s according to Eq. (2.33) in Ref. [2].
        
    References
    ----------
    [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
    [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)
    '''
    dx_G_map, dy_G_map = CFMH['dx_G'], CFMH['dy_G']
    
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
    beta0 = CFMH['beta0']
    dE_E = psigma*beta0**2 # = eta in Ref. [2].
    one_hateta2 = ((1 + dE_E)**2 - 1 + beta0**2)/beta0**2 # (1 + \hat \eta)**2; Eq. (2.17) in Ref. [2].
    kx, ky = CFMH['kx'], CFMH['ky']
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


class cfm(hard_edge_element):
    def __init__(self, components=[0], tilt=0, warn=True, **kwargs):
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
                        
        tilt: float, optional
            The tilt between the dipole component and the higher-order magnet components of the cfm.
            
        References
        ----------
        [1] M. Titze: "Approach to Combined-function magnets via symplectic slicing", Phys. Rev. STAB 19 054002 (2016)
        [2] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019)
        '''
        assert len(components) > 0
        self.components = components
        self.tilt = tilt
        hard_edge_element.__init__(self, warn=False, **kwargs)

        # also compute the overal bend angles, if a length has been provided:
        if hasattr(self, 'length'):
            self.phix = self.length*self.kx # r*phi = L; kx = 1/r
            self.phiy = self.length*self.ky
        else:
            if warn:
                warnings.warn('Length of dipole not specified. Bend angle can not be determined.')


    def setStyle(self, *args, **kwargs):
        '''
        Parameters
        ----------
        style: str, optional
            Name of the field in 'self', denoting the Hamiltonian to be used.
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
        self.hamiltonian = getattr(self, self._style)

    def calcHamiltonian(self, **kwargs):
        '''
        Calculate the Hamiltonian of the combined-function magnet.
        '''
        kwargs = self.__dict__
        kwargs['components'] = self.components
        kwargs['tilt'] = self.tilt
        cfmh = CFMHamiltonian(**kwargs)
        for k, v in cfmh.items():
            setattr(self, k, v)

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
        kwargs['components'] = components
        cfm.__init__(self, *args, **kwargs)

        
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
        