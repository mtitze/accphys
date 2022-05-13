from legacycontour import _cntr as cntr
import numpy as np
import matplotlib.pyplot as plt
from lieops.solver import getRealHamiltonFunction

def plot_contours(hamiltonian, xrange, yrange, plane=[0, 1], levels=[], q=[], p=[], **kwargs):
    '''
    Show the contours of the real-valued Hamiltonian.
    
    plane:
        define what components to show
    '''
    dim = hamiltonian.dim
    # here we do not use matplotlib contourf, instead an older library by which we can also separate the contour
    # from the plotting.
    assert len(q) == len(p)
    
    magr = getRealHamiltonFunction(hamiltonian)
    
    c1, c2 = plane
    def magr_plane(x, y):
        inp = [0]*2*dim
        inp[c1] = x
        inp[c2] = y
        return magr(*inp)
    
    X, Y = np.meshgrid(xrange, yrange)
    h = magr_plane(X, Y)
    
    if len(q) > 0 and len(p) > 0:
        for k in range(len(q)):
            levels += [magr_plane(q[k], p[k])]
    assert len(levels) > 0

    c = cntr.Cntr(X, Y, h)
    for level in levels:
        nlist = c.trace(level, level, 0)
        segs = nlist[:len(nlist)//2]

        for seg in segs:
            plt.plot(seg[:,0], seg[:,1], **kwargs)
    return plt