import os
from .common import madx2beamline

from lieops.solver.splitting import yoshida
from accphys import beamline

def test_example(lattice_file='xmpl1.madx', tol=1e-8, tol2=2e-6, tol3=1e-14, **kwargs):
    '''
    Test to import a MAD-X file and perform certain operations on a
    1D model, like splitting, hadamard, Magnus, ...
    '''
    lattice_file = f'{os.getcwd()}/tests/{lattice_file}'
    seq = madx2beamline(lattice_file=lattice_file, **kwargs)
    
    part1 = seq.copy()
    part1.setHamiltonians(0)
    part1.calcOneTurnMap(power=30, tol=tol)
    
    y1 = yoshida()
    yoshida_scheme = y1.build(0)

    step = 0.02
    keys = [(0, 2), (1, 1), (2, 0)]
    part2 = part1.split(keys=keys, scheme=yoshida_scheme, step=step)
    part2.calcOneTurnMap(power=10)
    
    bl_hdm = part2.hadamard(keys=keys, power=30)
    bl_hdm.calcOneTurnMap(power=30)
    
    bl_mag_A = beamline(bl_hdm[0].hamiltonian)
    bl_mag_B = bl_hdm[1:].magnus(order=6, time=False)
    bl_mag_A.calcOneTurnMap(method='2flow')
    bl_mag_B.calcOneTurnMap(method='channell', n_slices=100)
    bl_mag = bl_mag_A + bl_mag_B
    
    xi0, eta0 = 0.0027, -0.0012 # these values should not be too large
    
    p1 = part1(xi0, eta0)
    p2 = part2(xi0, eta0)
    p3 = bl_hdm(xi0, eta0)
    p4 = bl_mag(xi0, eta0)
        
    assert all([abs(p1[k] - p2[k]) < tol2 for k in range(2)])
    assert all([abs(p2[k] - p3[k]) < tol2 for k in range(2)])
    assert all([abs(p3[k] - p4[k]) < tol2 for k in range(2)])
    assert all([abs(p4[k] - p1[k]) < tol2 for k in range(2)])
    
    # Test time reversability
    part1_rev = part1[::-1]
    inp = [0.001, -0.0042]
    out1 = part1(*inp)
    out2 = part1_rev(*[o.conjugate() for o in out1]) # conjugate means going backwards in time, i.e. reverting the direction of the momenta.
    assert all([abs(out2[k] - inp[k]) < tol3 for k in range(2)])
    