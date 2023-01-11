import os
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
