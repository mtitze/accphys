import pytest
import numpy as np

from lieops import poly, create_coords
from lieops.solver.splitting import yoshida

from accphys import beamline
from accphys.elements import hard_edge_element

@pytest.mark.parametrize("x0, y0, px0, py0, gstr, length", [(3, 0.33, 1, 0.11, 0.94, 1.337), 
                                                            (3, 0.33, 1, 0.11, -0.878, 1.337)])
def test_quad_ref(x0, y0, px0, py0, gstr, length, tol=1e-14):
    '''
    Test final flow coordinates vs. expectation for a single normally aligned thick quadrupole.
    '''
    x, y, px, py = create_coords(2, real=True)
    ham_quad = 0.5*(px**2 + py**2) + 0.5*gstr*(x**2 - y**2)
    quad = hard_edge_element(ham_quad, length=length)
    b1 = beamline(quad)
    b1.calcOneTurnMap(power=40)
    b2 = beamline(quad)
    b2.calcOneTurnMap(method='heyoka')
        
    # expectation
    sg = np.sqrt(np.abs(gstr))
    cos = np.cos(sg*length)
    sin = np.sin(sg*length)
    cosh = np.cosh(sg*length)
    sinh = np.sinh(sg*length)
    if gstr > 0:
        xf = cos*x0 + 1/sg*sin*px0
        yf = cosh*y0 + 1/sg*sinh*py0
        pxf = -sg*sin*x0 + cos*px0
        pyf = sg*sinh*y0 + cosh*py0
    elif gstr < 0:
        xf = cosh*x0 + 1/sg*sinh*px0
        yf = cos*y0 + 1/sg*sin*py0
        pxf = sg*sinh*x0 + cosh*px0
        pyf = -sg*sin*y0 + cos*py0
    else: # kstr = 0
        xf = px0*length + x0
        yf = py0*length + y0
        pxf = px0
        pyf = py0
    
    xi1_0 = (x0 + px0*1j)/np.sqrt(2)
    xi2_0 = (y0 + py0*1j)/np.sqrt(2)
    eta1_0 = xi1_0.conjugate()
    eta2_0 = xi2_0.conjugate()
    
    for b in [b1, b2]:
        xi1f_1, xi2f_1, eta1f_1, eta2f_1 = b(xi1_0, xi2_0, eta1_0, eta2_0)
        xf_1 = (xi1f_1 + eta1f_1)/np.sqrt(2)
        yf_1 = (xi2f_1 + eta2f_1)/np.sqrt(2)
        pxf_1 = (xi1f_1 - eta1f_1)/np.sqrt(2)/1j
        pyf_1 = (xi2f_1 - eta2f_1)/np.sqrt(2)/1j
        
        assert abs(xf_1.imag) < tol
        assert abs(yf_1.imag) < tol
        assert abs(pxf_1.imag) < tol
        assert abs(pyf_1.imag) < tol

        assert abs(xf_1 - xf) < tol
        assert abs(yf_1 - yf) < tol
        assert abs(pxf_1 - pxf) < tol
        assert abs(pyf_1 - pyf) < tol

@pytest.mark.parametrize("xi0, eta0, yoshida_order, n_slices, tol", 
                         [(-0.0047, 0.0038, 0, 3, 1e-6), 
                          (0.022, 0.015, 0, 4, 5e-4), 
                          (-0.3, 0.53, 0, 12, 5e-3)])
def test_splitting1(xi0, eta0, yoshida_order, n_slices, tol):
    '''
    Test if splitting a single element gives roughly the same values in tracking as if
    passing through the original element.
    '''
    ham = poly(values={(1, 1): 0.5, (0, 2): -1/4, (2, 0): -1/4, (3, 0): -25.9,
                       (2, 1): -77.7, (1, 2): -77.7, (0, 3): -25.9})
    ele = hard_edge_element(ham, length=0.18)
    
    bl = beamline(ele)
    
    y1 = yoshida()
    yoshida_scheme = y1.build(yoshida_order)

    keys = [(0, 2), (1, 1), (2, 0)]
    bls = bl.split(keys=keys, scheme=yoshida_scheme, n_slices=n_slices)
    
    result = bl(xi0, eta0, method='heyoka')
    result_s = bls(xi0, eta0, method='heyoka')
    
    assert max([np.abs(result[k] - result_s[k]) for k in range(2)]) < tol
    
