import numpy as np
from latticeadaptor.core import LatticeAdaptor

from njet.jet import factorials

from .from_pandas import from_pandas

def file2dataframe(filename, **kwargs):
    '''
    Load MAD-X or Elegant lattice from file, using LatticeAdaptor module, 
    and construct a suitable dataframe object. Furthermore, return the
    required input parameters for to_beamline routine.
    
    Parameters
    ----------
    filename: str
        The name of the MAD-X lattice to be loaded.
        
    **kwargs
        Optional arguments passed to 'to_beamline' routine.
        
    Returns
    -------    
    Pandas dataframe
        A Pandas dataframe object, representing the loaded sequence of the lattice.
        
    dict
        A dictionary containing MAD-X specific input parameters for the general 'to_beamline' routine.
    '''
    supported_types = ['madx', 'lte']
    file_type = filename.split('.')[-1]
    if file_type not in supported_types:
        raise NotImplementedError(f"File of type '{file_type}' not supported.")
    
    la = LatticeAdaptor()
    la.load_from_file(filename, ftype=file_type)
    raw_df = la.table
    
    # MAD-X specific labels
    position_label='at'
    length_label='L'
    bend_kx_label = 'K0'
    angle_label = 'ANGLE'
    max_seek_order = 13 # maximal order of multipoles to be considered
    _default_position = 0.5 # The position of the elements relative to its center (0.5)

    component_indices = [j for j in range(max_seek_order) if f'K{j}' in raw_df.columns]
    component_labels = [f'K{j}' for j in component_indices]
    if bend_kx_label not in raw_df.columns and angle_label in raw_df.columns:
        # add kx (and K0 label) to the dataframe, computed from the bend angles
        angles = raw_df[angle_label].values
        lengths = raw_df[length_label].values 
        valid_indices = np.logical_and((~np.isnan(angles)), lengths > 0)
        kx = np.zeros(len(raw_df))
        kx[valid_indices] = angles[valid_indices]/lengths[valid_indices] # r*phi = L; kx = 1/r
        raw_df[bend_kx_label] = kx
        component_labels = [bend_kx_label] + component_labels
        component_indices = [0] + component_indices

    # drop elements with zero length and uneccesary columns;
    # N.B. E1 and E2 denote rotation angles of the pole-faces. If they are non-zero,
    # they are usually half the bend angle (in the rectangular case).
    columns_oi = [position_label, length_label] + component_labels + ['E1', 'E2']
    # if they exist, add skew-values to the components; TODO: check & verify this numerically
    for cl in component_labels:
        if cl + 'S' in raw_df.columns:
            raw_df[cl] = raw_df[cl].values + raw_df[cl + 'S'].values*1j
    raw_df = raw_df.loc[raw_df[length_label] > 0][columns_oi]
    
    # (Checked against MAD-X tracking: bend, quad, sextupole, ; factorials required)
    facts = factorials(max(component_indices))
    for j in range(len(component_indices)):
        index = component_indices[j]
        cl = component_labels[j]
        raw_df[cl] = raw_df[cl].values/facts[index]
        
    if len(component_labels) == 0: # special case: Only pure drifts exist
        raw_df[bend_kx_label] = [0]*len(raw_df)
        component_labels = [bend_kx_label]
        component_indices = [0]
        
    to_beamline_inp = {'component_labels': component_labels, 'component_indices': component_indices, 'position_label': position_label,
                    'length_label': length_label, 'position': kwargs.get('position', _default_position)}
    
    return raw_df, to_beamline_inp


def from_madx(filename, beta0, **kwargs):
    '''
    Load MAD-X or Elegant lattice from file and construct a beamline object from the data.
    
    Parameters
    ----------
    filename: str
        The name of the MAD-X lattice to be loaded.
        
    beta0: float
        The realtivistic beta-factor (related to the energy of the beam). This is required later to
        build the Hamiltonians.
        
    **kwargs
        Optional arguments passed to 'load_madx_file' routine.
        
    Returns
    -------
    beamline
        A beamline object representing the sequence of elements in the given lattice.
    '''
    raw_df, inp = file2dataframe(filename=filename, **kwargs)
    inp.update(kwargs)
    return from_pandas(raw_df, beta0=beta0, **inp)
