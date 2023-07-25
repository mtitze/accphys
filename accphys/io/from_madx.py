import numpy as np
from latticeadaptor.core import LatticeAdaptor

from njet.jet import factorials

from .from_pandas import from_pandas

def from_madx(filename, beta0, **kwargs):
    '''
    Load MAD-X or Elegant lattice from file, using LatticeAdaptor module, 
    and construct a suitable dataframe object. Furthermore, return the
    required input parameters for to_beamline routine.
    
    Parameters
    ----------
    filename: str
        The name of the MAD-X lattice to be loaded.

    beta0: float
        The realtivistic beta-factor (related to the energy of the beam). This is required later to
        build the Hamiltonians.
        
    Returns
    -------    
    beamline
    '''
    la = LatticeAdaptor()
    la.load_from_file(filename, ftype='madx') # 'lte' for elegant
    raw_df = la.table
    
    # MAD-X specific labels
    _default_position = 0.5 # The position of the elements relative to its center (center at 0.5)
    position_label='at'
    length_label='L'
    component_prefix = 'K'
    component_labels = [c for c in raw_df.columns if c[0] == component_prefix and c[1:].isnumeric()]
    bend_kx_label = 'K0'
    angle_label = 'ANGLE'
    if bend_kx_label not in component_labels and angle_label in raw_df.columns:
        # add the bend strengths to the dataframe, computed from the bend angles
        angles = raw_df[angle_label].values
        lengths = raw_df[length_label].values 
        valid_indices = np.logical_and((~np.isnan(angles)), lengths > 0)
        kx = np.zeros(len(raw_df))
        kx[valid_indices] = angles[valid_indices]/lengths[valid_indices] # r*phi = L; kx = 1/r
        raw_df[bend_kx_label] = kx
        component_labels = [bend_kx_label] + component_labels

    # If exist, add skew-values to the components; TODO: check & verify this numerically
    for cl in component_labels:
        if cl + 'S' in raw_df.columns:
            raw_df[cl] = raw_df[cl].values + raw_df[cl + 'S'].values*1j
            
    # E1 and E2 denote rotation angles of pole-faces. If they are non-zero, they are usually half the bend angle (in the rectangular case).
    columns_oi = [position_label, length_label] + component_labels + ['E1', 'E2']
    raw_df = raw_df.loc[raw_df[length_label] > 0][columns_oi] # every row containing length zero will be dropped
    
    # Factorials correction required
    component_indices = [int(cl[1:]) for cl in component_labels]
    facts = factorials(max(component_indices))
    for j in range(len(component_indices)):
        index = component_indices[j]
        cl = component_labels[j]
        raw_df[cl] = raw_df[cl].values/facts[index]
        
    if len(component_labels) == 0: # special case: Only pure drifts exist
        raw_df[bend_kx_label] = [0]*len(raw_df)
        component_labels = [bend_kx_label]
        component_indices = [0]


    return from_pandas(raw_df, beta0=beta0, position_label=position_label, length_label=length_label, component_indices=component_indices,
                       component_labels=component_labels, position=kwargs.get('position', _default_position), **kwargs)
