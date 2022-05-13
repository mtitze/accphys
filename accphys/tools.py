# Collection of some common routines, used to plot and perform standard tasks

def multiturn(bl, n_reps: int, xi0, post=lambda x: x):
    '''
    Perform tracking for a given number of repetitions.
    
    Parameters
    ----------
    bl: beamline, lieoperator or poly
        The object through which we want to track.
    
    n_reps: int
        The number of repetitions.
        
    xi0: The start vector xi0 = [xi0_1, xi0_2, ...]. By default: xi0_k = (q_k + i*p_k)/2, where q_k and
         p_k are the canonical coordinates in ordinary phase space.
    
    post: callable, optional
        An optional function to be applied after bl. 
        Example: We want to compute (A o bl o B)**n at xi0, where A = B**(-1) are maps to normal form.
        Then we need to record for Y0 := B(xi0) the values A(bl(Y0)), A(bl**2(Y0)) etc. Here A = post must
        be inserted.
    
    Returns
    -------
    list
        A list so that the k'th entry corresponds to the k-th component for the requested turns.
    '''
    dim = len(xi0)
    point = xi0
    points = [[] for c in range(dim)]
    for k in range(n_reps):
        point = bl(point)
        post_point = post(point)
        for c in range(dim):
            points[c].append(post_point[c])
    return points


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

def getPhaseAdvanceFunc(detuning_dict):
    '''
    Create a function giving the phase advance with respect to the action(s).
    
    Parameters
    ----------
    detuning_dict: dict
        The output of detuning routine.
        
    Returns
    -------
    callable
        A function taking values in *actions and s, returning the phase advance according to (e.g.)
        Ref. [1], Eq. (1.47b).
        
    Reference(s):
    [1]: M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019).
    '''
    dim = len(next(iter(detuning_dict.keys())))//2
    deps = [_depsilon(detuning_dict, k) for k in range(dim)]
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
