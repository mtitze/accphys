import os
import pytest
import numpy as np

from .common import madx2beamline, qp2xieta
from lieops.solver.splitting import yoshida, recursive_monomial_split
from accphys import beamline

lattice_file = f'{os.getcwd()}/tests/xmpl1.madx'
seq = madx2beamline(lattice_file=lattice_file)

def test_splitting(tol1=2e-10, tol2=5e-7, tol3=5e-15):
    '''
    Test various splitting methods on the beamline.
    '''
    part1 = seq.copy()
    part1.setProjection(0, 1)
    
    part2 = part1.split(n_slices=13)
    part3 = part1.split(n_slices=20, method=recursive_monomial_split, scheme=[0.5, 1, 0.5])
    
    xi0, eta0 = 0.00027, -0.00012
    xi1, eta1 = 0.0006, -0.0004
    
    r1 = part1(xi0, xi1, eta0, eta1, power=10)
    r2 = part2(xi0, xi1, eta0, eta1, power=10)
    r3 = part3(xi0, xi1, eta0, eta1, power=10)
    r4 = part3(xi0, xi1, eta0, eta1, method='channell')
    
    assert max(abs(np.array(r1) - np.array(r2))) < tol1
    assert max(abs(np.array(r1) - np.array(r3))) < tol2
    assert max(abs(np.array(r1) - np.array(r4))) < tol2
    assert max(abs(np.array(r3) - np.array(r4))) < tol3
    
def test_example(tol=1e-8, tol2=2e-6, tol3=1e-14):
    '''
    Test a typical MAD-X lattice (1D) with respect to tracking after
    various lattice operations.
    '''
    
    part1 = seq.copy()
    part1.setProjection(0)
    
    y1 = yoshida()
    yoshida_scheme = y1.build(0)

    step = 0.02
    keys = [(0, 2), (1, 1), (2, 0)]
    part2 = part1.split(keys=keys, scheme=yoshida_scheme, step=step)
    
    bl_hdm = part2.reshuffle(keys=keys, power=30)
    
    bl_mag_A = bl_hdm[:-1].magnus(order=6, time=False)
    bl_mag_B = beamline(bl_hdm[-1].hamiltonian)
    bl_mag_A.calcOneTurnMap(method='channell', n_slices=100)
    bl_mag_B.calcOneTurnMap(method='2flow')
    bl_mag = bl_mag_A + bl_mag_B
    
    xi0, eta0 = 0.0027, -0.0012 # these values should not be too large
    
    p1 = part1(xi0, eta0, power=30, tol=tol)
    p2 = part2(xi0, eta0, power=10)
    p3 = bl_hdm(xi0, eta0, power=30)
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
    
@pytest.mark.parametrize("q0, p0", [(0, 0)])
def test_dragtfinn(q0, p0, order=6, tol=5e-5, magnus_order=6):
    '''
    Test a typical MAD-X lattice (1D) in the following steps:
    1) Construct the Dragt/Finn factorization of the lattice.
    2) Performing a magnus-combination after reshuffling the original
       lattice, then expanding the result again using Dragt/Finn factorization.
    3) Compare the results of 1) and 2) in regards of their Hamiltonians and tracking results.
    
    N.B: Tests currently only working at zero; the TPSA appraoch is too rough for
         most checks. Requires dedicated test.
    '''
    df_inp = {'order': order, 'power': 30, 'pos2': 'left', 'tol': tol, 'comb2': False}
    
    part1 = seq.copy()
    part1.setProjection(0)
    
    xieta0 = qp2xieta(q0, p0)
    part1_df = part1.dragtfinn(*xieta0, **df_inp) # Dragt/Finn factorization of the lattice 'part1' around the point xieta0
    part1_df2 = part1_df.dragtfinn(*xieta0, **df_inp) # Repeat Dragt/Finn factorization; it must lead to the same lattice
    
    tolerances1 = [1e-15, 1e-15, 1e-12, 1e-9, 5e-6, 5e-3]
    assert len(part1_df) == len(part1_df2)
    assert all([(part1_df[k].hamiltonian - part1_df2[k].hamiltonian).above(tolerances1[k]) == 0 for k in range(len(tolerances1))])
    
    # Combine the higher-order factors of the Dragt/Finn factorization by Magnus-series.
    # The order of the preceeding Dragt/Finn factorization should be sufficiently high to obtain better accuracy
    bl_mag = part1_df[0:2] + part1_df[2:].magnus(order=magnus_order, max_power=10, time=False)
    
    # Expand bl_mag again:
    bl_mag_df = bl_mag.dragtfinn(*xieta0, **df_inp)
    
    # compare Hamiltonians of the two Dragt/Finn factorizations:
    tolerances2 = [1e-15, 1e-15, 5e-13, 1e-9, 5e-7, 5e-4]
    assert len(part1_df) == len(bl_mag_df)
    assert all([(part1_df[k].hamiltonian - bl_mag_df[k].hamiltonian).above(tolerances2[k]) == 0 for k in range(len(tolerances2))])
    
    # split the part1 lattice & perform reshuffling using Hadamard's Lemma:
    y1 = yoshida()
    yoshida_scheme = y1.build(0)
    step = 0.02
    keys = [(0, 2), (1, 1), (2, 0)]
    part2 = part1.split(keys=keys, scheme=yoshida_scheme, step=step)
    bl_hdm = part2.reshuffle(keys=keys, power=30)
    # check that the degrees of 'bl_hdm' are arranged as expected; note that
    # in the 1D-case it is possible to combine the 2nd order Hamiltonians into a single element:
    assert (e.hamiltonian.maxdeg() > 2 for e in bl_hdm[:-1]) and bl_hdm[-1].hamiltonian.maxdeg() == 2
    
    # combine the reshuffled lattice by Magnus series:
    bl_mag2 = bl_hdm[:-1].magnus(order=magnus_order, max_power=10, time=False) + bl_hdm[-1]
    bl_mag2.calcOneTurnMap(method='njet', n_slices=10, power=30)

    # expand both the reshuffled lattice and the newly lattice bl_mag2:
    bl_mag2_df = bl_mag2.dragtfinn(*xieta0, **df_inp)
    df_inp['tol'] = 2e-4 # relaxing the tolerance appears to not significantly change the next calculation, but may improve a bit the speed.
    bl_hdm_df = bl_hdm.dragtfinn(*xieta0, **df_inp)
    
    tolerances3 = [1e-15, 1e-15, 5e-13, 1e-9, 5e-7, 1.2e-4]
    assert len(bl_mag2_df) == len(bl_hdm_df)
    assert all([(bl_mag2_df[k].hamiltonian - bl_hdm_df[k].hamiltonian).above(tolerances3[k]) == 0 for k in range(len(bl_hdm_df))])    
    
    tolerances4 = [1e-14, 1e-14, 5e-4, 3e-3, 3e-3, 3e-3]
    # since the higher-order values differ more significantly, we use relative errors
    assert len(bl_mag2_df) == len(part1_df)
    for k in range(len(bl_mag2_df)):
        ak, bk = bl_mag2_df[k].hamiltonian, part1_df[k].hamiltonian
        assert ak.keys() == bk.keys()
        max_rel_error = 0
        for key in ak.keys():
            value1 = ak[key]
            value2 = bk[key]
            max_abs_value = max([abs(value1), abs(value2)])
            rel_error = abs(value1 - value2)/max_abs_value
            if rel_error > max_rel_error:
                max_rel_error = rel_error
        assert max_rel_error < tolerances4[k]
        