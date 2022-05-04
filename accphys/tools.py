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
    Get the detuning with amplitude factors for the given normal form dictionary which is returned by the bnf routine.
    
    Parameters
    ----------
    dict
        The output of the bnf routine.
    
    Returns
    -------
    list
        A list [f0, ..., fn], where n is the given order of the normal form. Every entry
        fj = [ej0, ... ejm] corresponds to a list of the coefficients in front of the actions, 
        where m is the underlying dimension.
    '''
    hks = nfdict['Hk']
    hk_last = hks[-1]
    # for every order j, the normalized Hamiltonian includes the power (xi*eta)**(j + 1)
    detuning = []
    for power in range(len(hks)):
        detuning_power = []
        for k in range(hk_last.dim):
            powers_oi = [0 if i != k else power + 1 for i in range(hk_last.dim)]
            detuning_power.append(hk_last[tuple(powers_oi + powers_oi)])
        detuning.append(detuning_power)
    return detuning

