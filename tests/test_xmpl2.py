import os
import mpmath as mp
import pytest

from lieops import lexp

from .common import madx2beamline
from accphys import beamline

lattice_file = f'{os.getcwd()}/tests/xmpl1.madx'
seq = madx2beamline(lattice_file=lattice_file)

def test_normalform1(tol1=5e-15, tol2=5e-11):
    '''
    Test the normal form procedure;
    Check if running the case with mpmath gives the same
    orientation.
    '''
    part1 = seq.copy()
    part1.setProjection(0)
    nfdict1 = part1.normalform(power=30, order=4, tol=1e-8)
    
    part2 = seq.copy()
    part2.setProjection(0)
    mp.mp.dps = 32
    part2.apply(mp.mpc)
    nfdict2 = part2.normalform(power=30, order=4, tol=1e-8)
    
    diff = sum(nfdict1['normalform']) - sum(nfdict2['normalform'])
    assert list(diff.keys()) == [(1, 1), (2, 2)]
    assert abs(diff[1, 1]) < tol1
    assert abs(diff[2, 2]) < tol2
    
def test_normalform2():
    '''
    Test if normalizing is an invariant operation.
    '''
    part1 = seq.copy()
    part1.setProjection(0)
    
    # Compute the normal form of the given sequence
    order = 6
    nfdict1 = part1.normalform(0, 0, order=order, power=30, tol=8e-10)
    nf1 = sum([n.above(1e-3) for n in nfdict1['normalform']])
    
    # The normal form must contain only powers of the actions, here up and including 3rd power (order=6)
    assert [(1, 1), (2, 2), (3, 3)] == list(nf1.keys())
    
    # Normalize the resulting beamline using the map(s) to normal form:
    dfn = [e.copy() for e in nfdict1['dragtfinn']]
    for chi in nfdict1['chi'][0:1]:
        dfn = lexp(chi)(*dfn, method='2flow') # for the 2nd-order polynomial, the exact (2flow) method can be used.
    for chi in nfdict1['chi'][1:]:
        dfn = lexp(chi)(*dfn, power=30)
    part1_n = beamline(*[lexp(e) for e in dfn])
    
    nfdict_n = part1_n.normalform(0, 0, order=order, power=40) # power should be sufficently high here.
    nf2 = sum([n.above(1e-3) for n in nfdict_n['normalform']])

    # check if both nf1 and nf2 are approximately equal
    assert [(1, 1), (2, 2), (3, 3)] == list(nf2.keys())
    diff = abs(nf1 - nf2)
    assert diff[1, 1] < 1e-14
    assert diff[2, 2] < 1e-9
    assert diff[3, 3] < 0.15
    
@pytest.mark.parametrize("xi0, eta0, tol", [(-0.0004, 0.001, 2e-7), 
                                            (-0.00068, -0.0001*1j + 0.0001, 1e-8)])
def test_normalform3(xi0, eta0, tol, order=6):
    '''
    Test tracking through the normal form vs. the normalized map.
    '''
    part1 = seq.copy()
    part1.setProjection(0)
    
    nfdict1 = part1.normalform(0, 0, order=order, power=30, tol=8e-10)
    
    nf = beamline(*[lexp(c) for c in nfdict1['chi'][::-1]])
    nfi = beamline(*[lexp(-c) for c in nfdict1['chi']])
    
    ref = nf + part1 + nfi
    nf_beamline = beamline(lexp(sum(n for n in nfdict1['normalform'])))
    
    point1 = ref(xi0, eta0, power=30)
    point2 = nf_beamline(xi0, eta0, power=30)
    
    assert all([abs(point1[k] - point2[k]) < tol for k in range(2)])
    