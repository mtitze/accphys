import pytest
import numpy as np
from scipy.linalg import expm

from lieops import create_coords, lexp
from lieops.core.tools import poly2ad

from accphys.elements import cfm, quadrupole, polefaceRM

'''
    Reference(s):
    [1] M. Titze: "Space Charge Modeling at the Integer Resonance for the CERN PS and SPS", PhD Thesis (2019).
'''


@pytest.mark.parametrize("kx, ky, g, h, beta0",
                         [(0.321, -2.1, 0, 5.2, 0.753),
                          (0.42, 2.21, -0.4616, -5.32, 0.863)])
def test_cfm(kx, ky, g, h, beta0, tol=1e-15):
    '''
    Test a CFM element against Eq. (2.57) in Ref. [1].    
    '''
    # Note that we have to insert h/2 to compare the result with those from the thesis
    current_cfm = cfm(beta0=beta0, components=[kx - ky*1j, g, h/2])
    
    # Comparison of Eq. (2.57) in Thesis
    x, y, sigma, px, py, psigma = create_coords(3, real=True)

    kappa_at_thesis = y**4*kx**2*g/24 - x**2*kx**2/2 + x*y**3*g*kx*ky/6 - x*y*kx*ky \
    - x**3*kx*g/3 + x*y**2*kx*g/2 - x**4*kx*h/8 + x**2*y**2*kx*h/2 \
    - y**4*kx*h/24 - x*kx - y**4*ky**2*g/24 - y**2*ky**2/2 + y**3*ky*g/6 + x*y**3*ky*h/6 \
    - y*ky - x**2*g/2 + y**2*g/2 - x**3*h/6 + y**2*x*h/2 # G = kappa*e/p0*A_t in = (1 + Kx*x + Ky*y)*e/p0*A_t in Thesis
    
    diff = current_cfm.G - kappa_at_thesis
    assert max([abs(v) for v in diff.values()]) < tol
    
def test_nodipole(cl=5, beta0=0.995, cfmlength=1, **kwargs):
    '''
    Test a CFM element in case that the dipole component is absent. In this case it must hold for the
    g_{j, k} elements (see Ref. [1]) for k >= 1:
    
    (k + 1) g_{k + 1, 0} == -2**k*lambda[k],
    
    where lambda[k] are the components of the CFM.
    '''
    components = [0] + list(np.random.rand(cl) + np.random.rand(cl)*1j)

    testele = cfm(components=components, length=cfmlength, beta0=beta0, **kwargs)
    testele = testele.project(0)
    
    gc = testele.g
    assert gc[(0, 0)] == 0
    assert gc[(1, 0)] == 0
    assert gc[(1, 1)] == 0
    for k in range(cl):
        assert gc[(k + 1, 0)] == -2**k/(k + 1)*components[k]
        assert gc[(k + 1, k + 1)] == gc[(k + 1, 0)].conjugate()
        for j in range(1, k):
            assert gc[(k, j)] == 0
        
@pytest.mark.parametrize("gstr", [(0.22321), (0.42), (-4.22)])
def test_quad(gstr, beta0=0.9664):
    '''
    In case of a quadrupole, it must hold
    G = -g/2*(x**2 - y**2),
    where g is the quad field strenght and G according to Ref. [1]. This is tested here.
    '''
    quad = quadrupole(gstr, beta0=beta0)
    x, y, z, px, py, pz = create_coords(dim=3, real=True)
    
    assert (x**2 - y**2)*(-gstr/2) == quad.G
    assert -gstr*x == quad.dx_G
    assert gstr*y == quad.dy_G
    

@pytest.mark.parametrize("rho, phi, inp", [(-24.57687739223107, -0.0033161251, [0.1, -2.2, 0.42, 0.62, 1.1, -0.23])])
def test_polefaceRM(rho, phi, inp, tol=1e-15):
    '''
    Test pole-face rotation matrix element; we test here
    if the matrix applied to some point yields the same
    result as the calculated flow from the underlying Lie-
    operator.
    '''
    edge = polefaceRM(rho=rho, phi=phi)
    ref0 = edge._matrix_xieta@np.array(inp)
    out1 = np.array(lexp(-edge.hamiltonian)(*inp, power=2)) # low power is sufficient because higher-order terms in the exponential series are zero here
    mat = expm(poly2ad(-edge.hamiltonian).transpose())
    out2 = mat@np.array(inp)
    out3 = edge(*inp, power=2)
    
    assert max(np.abs(ref0 - out1)) < tol
    assert max(np.abs(ref0 - out2)) < tol
    assert max(np.abs(ref0 - out3)) < tol
    
