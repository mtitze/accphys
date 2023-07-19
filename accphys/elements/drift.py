from lieops import create_coords, construct

from accphys.tools import energy2beta0

from .common import hard_edge_element

def DriftHamiltonian(sqrtexp: int=2, tol_drop=0, **kwargs):
    '''
    Compute the Hamiltonian of a drift.
    For the underlying coordinate system and further details see Refs. [1, 2] below.
    In particular, the Hamiltonian is given by Eq. (2.16) in Ref. [2].

    Parameters
    ----------
    energy or beta0: float
        Energy (in GeV) or beta0 = v/c of the particle in question.
    
    sqrtexp or power: int, optional
        Power up to which the square root of the drift should be expanded.
        
    tol_drop: float, optional
        Threshold below which terms in the Hamiltonian are considered to be zero.

    Returns
    -------
    dict
    '''
    # User input handling
    if 'energy' in kwargs.keys():
        beta0 = energy2beta0(kwargs['energy'], particle_type=kwargs.get('particle_type', 'electron'))
    elif not 'beta0' in kwargs.keys():
        raise RuntimeError("'energy' or 'beta0' parameter required.")
    else:
        beta0 = kwargs['beta0']
    # Set max_power default to be provided to (possible) Lie-operator arguments (see lieops.core.poly)
    _ = kwargs.setdefault('max_power', 10)

    
    assert 0 < beta0 and beta0 < 1
    # Compute the CFM drift part
    x, y, sigma, px, py, psigma = create_coords(3, real=True, **kwargs)
    one_hateta2 = lambda ps: ((1 + ps*beta0**2)**2 - 1 + beta0**2)/beta0**2 # Eqs. (15c) and (17) in Ref. [1]
    sqrt = lambda p1, p2, ps: (one_hateta2(ps) - p1**2 - p2**2)**(1/2)
    _ = kwargs.pop('power', None) # n.b. this will not remove the key in any calling instance
    drift_s = construct(sqrt, px, py, psigma, power=sqrtexp, **kwargs).above(tol_drop) # expand sqrt around [px, py, psigma] = [0, 0, 0] up to power.
    hamiltonian = psigma - drift_s
    hamiltonian = hamiltonian.pop((0, 0, 0, 0, 0, 0), None).above(tol_drop)
    prop = {}
    prop['beta0'] = beta0
    prop['sqrtexp'] = sqrtexp
    prop['dE_E'] = psigma*beta0**2
    prop['drift'] = hamiltonian
    prop['hamiltonian'] = hamiltonian
    prop['drift_sqrt'] = drift_s
    prop['coords'] = x, y, sigma, px, py, psigma
    return prop

