import pytest
import os
import numpy as np
from accphys.io import from_madx

from .common import madx2beamline

def test_import(lattice_file='xmpl1.madx', **kwargs):
    '''
    Test to import and convert a MAD-X file.
    '''
    lattice_file = f'{os.getcwd()}/tests/{lattice_file}'
    try:
        seq = madx2beamline(lattice_file=lattice_file, **kwargs)
    except:
        raise RuntimeError('Test failed.')
        
        
madx_file = f'{os.getcwd()}/tests/edge_test_lattice.madx'

@pytest.mark.parametrize("madx_file", [madx_file])
def test_edges(madx_file, tol1x=3e-5, tol1px=6e-6, tol2x=4e-5, tol2px=8e-6):
    '''
    Test if conversion from MAD-X works and actually yields the same results
    as tracking with the accphys module.
    
    A MAD-X lattice file is loaded with - and without - taking into account the dipole
    edges from rectangular bends. The tracking result is compared to those obtained
    in MAD-X, where MAD-X has been run using the 'edge_test.madx' file (see the
    comments there).    
    '''
    
    bl1 = from_madx(madx_file)
    bl1_0 = bl1.project(0)
    
    bl2 = from_madx(madx_file, disable_edges=True)
    bl2_0 = bl2.project(0)    
    
    x1, px1 = bl1_0.track(3.2e-3, 0, turns=200, real=True, power=10, output_format='coords')
    x2, px2 = bl2_0.track(2e-3, 0, turns=200, real=True, power=10, output_format='coords')
    
    madx_out_1 = f'{os.getcwd()}/tests/track_edges_true.obs0001.p0001' # MAD-X tracking data with edge effects
    madx_data_1 = np.loadtxt(madx_out_1, comments=['@', '*', '$'])
    
    madx_out_2 = f'{os.getcwd()}/tests/track_edges_false.obs0001.p0001' # MAD-X tracking data without edge effects
    madx_data_2 = np.loadtxt(madx_out_2, comments=['@', '*', '$'])

    assert max(np.abs(madx_data_1[1:, 2] - np.array(x1).real)) < tol1x
    assert max(np.abs(madx_data_1[1:, 3] - np.array(px1).real)) < tol1px
    
    assert max(np.abs(madx_data_2[1:, 2] - np.array(x2).real)) < tol2x
    assert max(np.abs(madx_data_2[1:, 3] - np.array(px2).real)) < tol2px

