import os
import mpmath as mp
import pytest

from lieops import lexp

from .common import madx2beamline
from accphys import beamline

lattice_file = f'{os.getcwd()}/tests/xmpl1.madx'
seq = madx2beamline(lattice_file=lattice_file)

def test_normalform1(tol1=5e-15, tol2=5e-10, tol3=1e-8):
    '''
    Test the normal form procedure;
    Check if running the case with mpmath gives the same
    orientation.
    '''
    part1 = seq.project(0)
    _ = part1.taylor_map(power=30, order=4, tol=1e-8)
    nfdict1 = part1.normalform(power=30, order=3)
    
    part2 = seq.project(0)
    mp.mp.dps = 32
    part2.apply(mp.mpc)
    _ = part2.taylor_map(power=30, order=4, tol=1e-8)
    nfdict2 = part2.normalform(power=30, order=3)

    n1, n2 = sum(nfdict1['normalform']).above(tol3), sum(nfdict2['normalform']).above(tol3)
    diff = n1 - n2
    assert list(diff.keys()) == [(1, 1), (2, 2)]
    assert abs(diff[1, 1]) < tol1
    assert abs(diff[2, 2]) < tol2
    
def test_normalform2():
    '''
    Test if normalizing is an invariant operation.
    '''
    part1 = seq.project(0)
    
    # Compute the normal form of the given sequence
    order = 6
    _ = part1.taylor_map(0, 0, order=order, power=30, tol=8e-10)
    nfdict1 = part1.normalform(power=30, order=order - 1)
    nf1 = sum([n.above(5e-2) for n in nfdict1['normalform']])
    
    # The normal form must contain only powers of the actions, here up and including 3rd power (order=6)
    assert [(1, 1), (2, 2), (3, 3)] == list(nf1.keys())
    
    # Normalize the resulting beamline using the map(s) to normal form:
    dfn = [e.copy() for e in nfdict1['dragtfinn']]
    for chi in nfdict1['chi'][0:1]:
        dfn = lexp(chi)(*dfn, method='2flow') # for the 2nd-order polynomial, the exact (2flow) method can be used.
    for chi in nfdict1['chi'][1:]:
        dfn = lexp(chi)(*dfn, power=30)
    part1_n = beamline(*[lexp(e) for e in dfn])
    
    _ = part1_n.taylor_map(0, 0, order=order, power=40)
    nfdict_n = part1_n.normalform(power=40, order=order - 1) # power should be sufficently high here.
    nf2 = sum([n.above(1e-3) for n in nfdict_n['normalform']])

    # check if both nf1 and nf2 are approximately equal
    assert [(1, 1), (2, 2), (3, 3)] == list(nf2.keys())
    diff = abs(nf1 - nf2)
    assert diff[1, 1] < 1e-14
    assert diff[2, 2] < 5e-9
    assert diff[3, 3] < 0.15
    
@pytest.mark.parametrize("xi0, eta0, tol0, xi1, eta1, tol1", 
                         [(-0.0004, 0.001, 2e-7, 0.00054, -0.00024, 1.5e-6), 
                          (-0.00068, -0.0001*1j + 0.0001, 1e-8, -0.00068, -0.0001*1j + 0.0001, 1.4e-6)])
def test_normalform3(xi0, eta0, tol0, xi1, eta1, tol1, order=6):
    '''
    Test tracking through the normal form vs. the normalized map.
    '''
    part1 = seq.project(0)
    
    _ = part1.taylor_map(0, 0, order=order, power=30, tol=8e-10)
    nfdict1 = part1.normalform(power=30, order=order)
    
    nf = nfdict1['N']
    nfi = nfdict1['Ni']
    nf_beamline = nfdict1['normalbl']

    test1 = nf + part1 + nfi
    point1 = test1(xi0, eta0, power=30)
    point2 = nf_beamline(xi0, eta0, power=30)
    assert all([abs(point1[k] - point2[k]) < tol0 for k in range(2)])
    
    test2 = nfi + nf_beamline + nf
    point3 = test2(xi1, eta1, power=30)
    point4 = part1(xi1, eta1, power=30)
    assert all([abs(point3[k] - point4[k]) < tol1 for k in range(2)])
    