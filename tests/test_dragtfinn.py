from lieops import poly
from lieops.linalg.bch import bch_2x2
from lieops.core.tools import ad2poly, poly2ad

from accphys import beamline

def test_df1D_order1(tol1=5e-12, tol2=1e-14, tol_checks=1e-11, max_power=10):
    '''
    Test if the 2x2 BCH formula + Dragt/Finn factorization of a beamline will give back the same values.
    '''
    bl = beamline(poly(values={(1, 1): 2.0627087458381115}, max_power=max_power), 
                  poly(values={(2, 0): 0.26886341802816816 - 0.5017539238372174*1j, 
                               (0, 2): 0.26886341802816816 + 0.5017539238372174*1j}, max_power=max_power))
    
    A = bl[0].hamiltonian
    B = bl[1].hamiltonian
    # Note that A@B != B@A in this example
    
    C = ad2poly(bch_2x2(poly2ad(B), poly2ad(A)), max_power=max_power) # Note the exchange in the order of A and B: First the A-operator must be executed, then B.
    bl_C = beamline(C)

    _ = bl_C.taylor_map(0, 0, order=1, power=30)
    cdf1 = bl_C.dragtfinn(power=10, comb2=False, tol_checks=tol_checks)
    cdf2 = bl_C.dragtfinn(method='2flow', comb2=False, tol_checks=tol_checks)
    
    for k in range(2):
        assert (cdf1[k].hamiltonian - bl[k].hamiltonian).above(tol1) == 0
        assert (cdf2[k].hamiltonian - bl[k].hamiltonian).above(tol2) == 0
        