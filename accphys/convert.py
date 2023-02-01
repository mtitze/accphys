from .beamline import beamline
from .elements import cfm

from tqdm import tqdm

from latticeadaptor.core import LatticeAdaptor
import numpy as np
from njet.jet import factorials

def prepare_df(hdf, position_label='s', length_label='L', position=0, tol=1e-6, **kwargs):
    '''
    Prepare a given data frame so that there will be no zero-spaces in between two elements.
    
    Parameters
    ----------
    hdf: pd.Dataframe
        Pandas dataframe object defining the beam sequence. The dataframe must contain the
        position_label and length_label columns.
        
    position_label: str, optional
        Label denoting the position of the individual elements within the dataframe.
        
    length_label: str, optional
        Label denoting the lengths of the individual elements within the dataframe.
    
    position: float, optional
        Defines the alignment of the elements relative to their actual position.
        Take a value between 0 and 1, where 0 means the position is given relative to the element start,
        while 1 means that they are taken with respect to the element end.
    
    tol: float, optional
        Tolerance below which we consider the end/start of two adjacent elements to agree with each other.
        
    Returns
    -------
    Pandas dataframe
        A dataframe object in which all elements are positioned so that there are no gaps inbetween (drift
        elements are added accordingly).
    '''
    
    toStartPos = lambda z, length: z - length*position # transform position to start of element.
    
    # ensure the Dataframe index is reset
    hdf = hdf.reset_index(drop=True)
    
    column_labels = list(hdf.columns)
    position_label_index = column_labels.index(position_label)
    length_label_index = column_labels.index(length_label)
    
    # Iterate through the entire sequence to check the position and lengths; insert drifts if necessary
    for k in range(len(hdf)):
        current_row = hdf.iloc[k]
        pos_1 = current_row[position_label]
        length = current_row[length_label]
        pos = toStartPos(pos_1, length)
        
        if k + 1 < len(hdf):
            next_row = hdf.iloc[k + 1]
            next_pos_1 = next_row[position_label]
            next_length = next_row[length_label]
            next_pos = toStartPos(next_pos_1, next_length)
        else:
            next_pos = pos + length

        if pos + length > next_pos + tol:
            # in this case the next element is overlapping the previous element due to a too large length.
            raise RuntimeError(f"Element at row-index {k}, covering [{pos}, {pos + length}] appears to overlap its successor, beginning at {next_pos}. Check lattice input and/or adjust position: {position} and/or tol: {tol}.")

        if pos + length + tol < next_pos:
            # here we have to insert additional drift space in between.
            empty_space = next_pos - pos - length
            
            new_row = [0]*len(hdf.columns)
            new_row[position_label_index] = pos + length
            new_row[length_label_index] = empty_space
            hdf.loc[k + 0.5] = new_row # insert a new row in between k and k + 1
            
    return hdf.sort_index().reset_index(drop=True)


def to_beamline(hdf, component_labels, component_indices, position_label='s', length_label='L', **kwargs):
    '''
    Construct a beamline from a given lattice.
    
    Parameters
    ----------
    hdf: Pandas dataframe
        A Pandas dataframe object containing the position, lengths and field strengths of the individual
        elements in the beamline.
                
    component_labels
        A list of floats, declaring the names of the columns in which to find the combined-function components.
        
    component_indices
        A list of integers, where the k-th entry denotes the index of the respective component.
        
    position_label: str, optional
        Label denoting the position of the individual elements within the dataframe.
        
    length_label: str, optional
        Label denoting the lengths of the individual elements within the dataframe.
        
    **kwargs
        Keyworded arguments passed to the cfm element.
        
    Returns
    -------
    beamline
        A beamline object representing the sequence of elements.
    '''
    assert len(component_labels) > 0
    assert len(component_labels) == len(component_indices)
    # Preparation; ensure that no empty space exists between elements (they will be filled with drifts if necessary):
    hdf = prepare_df(hdf, position_label=position_label, length_label=length_label, **kwargs)
            
    # group the given elements with respect to the remaining labels & find the uniques among them
    group_labels = [c for c in hdf.columns if c != position_label] # labels by which we distinguish different elements; custom columns may be added to artifically distinguish different element groups
    group_index_label = 'group_index'
    grp = hdf.groupby(group_labels, sort=False)
    hdf[group_index_label] = grp.ngroup()
    unique_elements = hdf.drop_duplicates(group_index_label)
    
    # build the elements:
    elements = []
    group_index = 0
    n_components = max(component_indices) + 1 # + 1 because of the 0th-component
    for n in tqdm(range(len(unique_elements)), disable=kwargs.get('disable_tqdm', False)):
        row = unique_elements.iloc[n]
        components = [0]*n_components
        for j in range(len(component_indices)):
            index = component_indices[j]
            label = component_labels[j]
            components[index] = row.get(label, 0)
        length = row.get(length_label, 0)
        assert group_index == row[group_index_label] # verify that the position in the element list corresponds to the group index given by ngroup.
        elements.append(cfm(components=components, length=length, **kwargs))
        group_index += 1
        
    # set the ordering
    ordering = list(hdf[group_index_label])
            
    return beamline(*elements, ordering=ordering)


def madx2dataframe(filename, **kwargs):
    '''
    Load MAD-X lattice from file, using LatticeAdaptor module, 
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
    la = LatticeAdaptor()
    la.load_from_file(filename, ftype='madx')
    raw_df = la.table
    
    # MAD-X specific labels
    position_label='at'
    length_label='L'
    bend_kx_label = 'K0'
    angle_label = 'ANGLE'
    max_seek_order = 13 # maximal order of multipoles to be considered
    madx_default_position = 0.5 # MAD-X tends to denote the position of the elements in the center

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
    # they are usually half the bend angle (in the rectangular case). We will ignore rectangular
    # bends for the time being and use s-bends here.
    columns_oi = [position_label, length_label] + component_labels
    # if they exist, add skew-values to the components; TODO: check & verify this numerically
    for cl in component_labels:
        if cl + 'S' in raw_df.columns:
            raw_df[cl] = raw_df[cl].values + raw_df[cl + 'S'].values*1j
    raw_df = raw_df.loc[raw_df[length_label] > 0][columns_oi]
    
    # (TO BE CHECKED; TODO)
    facts = factorials(len(component_labels))
    j = 0
    for cl in component_labels:
        raw_df[cl] = raw_df[cl].values/facts[j]
        j += 1
        
    if len(component_labels) == 0: # special case: Only pure drifts exist
        raw_df[bend_kx_label] = [0]*len(raw_df)
        component_labels = [bend_kx_label]
        component_indices = [0]
        
    to_beamline_inp = {'component_labels': component_labels, 'component_indices': component_indices, 'position_label': position_label,
                    'length_label': length_label, 'position': kwargs.get('position', madx_default_position)}
    
    return raw_df, to_beamline_inp


def madx2beamline(filename, beta0, **kwargs):
    '''
    Load MAD-X lattice from file and construct a beamline object from the data.
    
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
    raw_df, to_beamline_inp = madx2dataframe(filename=filename, **kwargs)
    to_beamline_inp.update(kwargs)
    return to_beamline(raw_df, beta0=beta0, **to_beamline_inp)
