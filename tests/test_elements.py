import pytest

from accphys.elements import *

@pytest.mark.parametrize("kx,ky,g,h,beta0",
                         [(0.321, -2.1, 0, 5.2, 0.753),
                          (0.42, 2.21, -0.4616, -5.32, 0.863)])
def test_cfm(kx, ky, g, h, beta0, tol=1e-15):
    '''
    Test a CFM element against Eq. (2.57) in Ref. [1].
    
    Reference(s):
    [1] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019).
    '''
    # Note that we have to insert h/2 to compare the result with those from the thesis
    current_cfm = cfm(beta0=beta0, components=[kx - ky*1j, g, h/2, 0])
    
    # Comparison of Eq. (2.57) in Thesis
    x, y, sigma, px, py, psigma = create_coords(3, real=True)

    kappa_at_thesis = y**4*kx**2*g/24 - x**2*kx**2/2 + x*y**3*g*kx*ky/6 - x*y*kx*ky \
    - x**3*kx*g/3 + x*y**2*kx*g/2 - x**4*kx*h/8 + x**2*y**2*kx*h/2 \
    - y**4*kx*h/24 - x*kx - y**4*ky**2*g/24 - y**2*ky**2/2 + y**3*ky*g/6 + x*y**3*ky*h/6 \
    - y*ky - x**2*g/2 + y**2*g/2 - x**3*h/6 + y**2*x*h/2 # G = kappa*e/p0*A_t in = (1 + Kx*x + Ky*y)*e/p0*A_t in Thesis
    
    diff = current_cfm._prop['G'] - kappa_at_thesis
    assert max([abs(v) for v in diff.values()]) < tol
