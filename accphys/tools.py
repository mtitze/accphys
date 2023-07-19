# Collection of some common routines, used to plot and perform standard tasks
import numpy as np
from scipy import constants

def energy2beta0(gev, particle_type='electron'):
    # Convert energy in GeV to beta0 (for electrons)
    energy_joule = gev*1e9*constants.electron_volt
    if particle_type == 'electron':
        m = constants.electron_mass
    elif particle_type == 'proton':
        m = constants.proton_mass
    else:
        raise RuntimeError('Particle type not understood.')
    c = constants.speed_of_light
    return np.sqrt(1 - m**2*c**4/energy_joule**2)

def f_identity(*x):
    return x

def f_compose(f, g):
    return lambda *x: f(*g(*x))

def detuning(nfdict):
    '''
    Get the detuning with amplitude terms for the given normal form dictionary which is returned by the bnf routine.
    
    Parameters
    ----------
    dict
        The output of the bnf routine.
    
    Returns
    -------
    dict
        A dictionary mapping the powers of (xi, eta)-coordinates to the coefficients in the Hamiltonian.
        These powers are equal in the sense that the resulting expression corresponds to a polynomial in
        the actions epsilon_k = xi_k*eta_k.
    '''
    hks = nfdict['Hk']
    hk_last = hks[-1]
    dim = hk_last.dim
    nforder = nfdict['order']
    # nforder: The order of the current normal form: Note that for every order j the 
    # normalized Hamiltonian includes the power (xi*eta)**(j + 1), while the homogeneous polynomials
    # reach up to power j + 2.
    out = {}
    for k, v in hk_last.items():
        k1, k2 = k[:dim], k[dim:]
        if k1 == k2 and sum(k1) <= nforder + 1: 
            # The first condition ensures that the term depends only on the actions.
            # The second condition ensures that we do not include too many powers which may 
            # be beyond the given normal form order.
            out[tuple(k1 + k1)] = v
    return out

def _depsilon(DH, k):
    '''
    Derive a given action-polynomial, represented by a dictionaly-like object,
    with respect to the k-th action. 
    
    Parameters
    ----------
    Hnf: poly or dict
        A polynomial or dict representing the powers depending only on its actions xi_k*eta_k.
        
    k: integer
        Index with respect to which we want to derive.
    '''
    dim = len(next(iter(DH.keys())))//2
    assert k <= dim - 1, 'Requested index of derivative direction larger than given dimension.'
    D = {}
    for j, v in DH.items():
        j1, j2 = j[:dim], j[dim:]
        assert j1 == j1, 'Given polynomial does not purely depend on its actions.'
        if j1[k] == 0:
            continue
        new = [jm for jm in j1]
        new[k] = j1[k] - 1
        D[tuple(new + new)] = v*j1[k]
    return D

def getPhaseAdvanceFunc(*deps):
    '''
    Create a function giving the phase advance with respect to the action(s).
    
    Parameters
    ----------
    deps
        The output of the _depsilon routine for each direction.
        Example: deps = *[_depsilon(detuning_dict, k) for k in range(dim)]

    Returns
    -------
    callable
        A function taking values in *actions and s, returning the phase advance according to (e.g.)
        Ref. [1], Eq. (1.47b).
        
    Reference(s):
    [1]: M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019).
    '''
    dim = len(deps)
    def pa(*action, s: float=1):
        '''
        Compute the phase advance in normal form.
        
        Parameters
        ----------
        *action
            The action(s) at which to obtain the phase advance.
            
        s: float, optional
            The free parameter defining the phase advance.
            
        Returns
        -------
        list
            A list where the k'th entry corresponds to the detuning of the k'th phase advance.
        '''
        # Example for dim = 2: deps[k].keys() = (0, 1, 0, 1), (1, 2, 1, 2) , ... for k = 0
        out = []
        for k in range(dim):
            sumk = 0
            for p, v in deps[k].items():
                prod = 1
                for j in range(dim):
                    if p[j] == 0: # Ensures that 0**0 = 1 so that e.g. (0, 0) (the constant term) is included
                        continue
                    prod *= action[j]**p[j]
                sumk += prod*v
            out.append(sumk*s)
        return out
    return pa

def get_optics(bl_optics, tol=1e-11):
    '''
    Obtain optics functions from the driving terms of a lattice.
    '''
    dterms = bl_optics
    optics = {}
    optics['betax'] = []
    for pos in range(len(dterms)):
        Sinv_pos = dterms[pos]['bnfout']['nfdict']['Sinv']
        if tol > 0: # consistency check if Sinv_pos is purely imaginary
            check = np.linalg.norm(Sinv_pos.imag)
            assert check < tol, f'Matrix at {pos} not purely real: {check} >= {tol} (tol); point unstable?'
        s00 = Sinv_pos.real[0, 0]
        optics['betax'].append(1/s00**2)
        
    return optics