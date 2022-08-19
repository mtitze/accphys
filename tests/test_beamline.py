import pytest
import numpy as np

from lieops import poly
from lieops.solver.yoshida import symint

from accphys import beamline
from accphys.elements import hard_edge_element

@pytest.mark.parametrize("q0, p0, yoshida_order, n_slices, tol", 
                         [(-0.0047, 0.0038, 0, 3, 1e-6), 
                          (0.022, 0.015, 1, 4, 5e-4), 
                          (-0.3, 0.53, 2, 12, 5e-3)])
def test_splitting1(q0, p0, yoshida_order, n_slices, tol):
    '''
    Test if splitting a single element gives roughly the same values in tracking as if
    passing through the original element.
    '''
    ham = poly(values={(1, 1): 0.5, (0, 2): -1/4, (2, 0): -1/4, (3, 0): -25.9,
                       (2, 1): -77.7, (1, 2): -77.7, (0, 3): -25.9})
    ele = hard_edge_element(ham, length=0.18)
    
    bl = beamline(ele)
    bl.calcOneTurnMap(method='heyoka')
    
    y1 = symint()
    yoshida_scheme = y1.build(yoshida_order)

    keys = [(0, 2), (1, 1), (2, 0)]
    bls = bl.split(keys=keys, scheme=yoshida_scheme, n_slices=n_slices)
    bls.calcOneTurnMap(method='heyoka')
    
    result = bl(q0, p0)
    result_s = bls(q0, p0)
    
    assert max(np.abs(result - result_s)) < tol