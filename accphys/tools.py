

# collection of the most common routines which are used to plot and perform certain checks

def multiturn(bl, n_reps: int, xi0):
    '''
    Perform tracking for a given number of repetitions.
    
    Parameters
    ----------
    bl: beamline, lieoperator or poly
        The object through which we want to track.
    
    n_reps: int
        The number of repetitions.
        
    xi0: The start vector; xi0 = [xi0_1, xi0_2, ...] so that xi0_k = (q_k + i*p_k)/2 holds.
    
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
        for c in range(dim):
            points[c].append(point[c])
    return points