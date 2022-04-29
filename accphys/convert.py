from .beamline import beamline
from .elements import cfm
from tqdm.auto import tqdm

from latticeadaptor.core import LatticeAdaptor
import numpy as np

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


def to_beamline(hdf, beta0, component_labels, position_label='s', length_label='L', **kwargs):
    '''
    Construct a beamline from a given lattice.
    
    Parameters
    ----------
    hdf: Pandas dataframe
        A Pandas dataframe object containing the position, lengths and field strengths of the individual
        elements in the beamline.
        
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
    # Preparation; ensure that no empty space exists between elements (they will be filled with drifts if necesary):
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
    for n in tqdm(range(len(unique_elements))):
        row = unique_elements.iloc[n]
        components = [row[c] for c in component_labels]
        length = row[length_label]
        assert group_index == row[group_index_label] # verify that the position in the element list corresponds with the group index given by ngroup.
        elements.append(cfm(beta0=beta0, components=components, length=length, **kwargs))
        group_index += 1
        
    # set the ordering
    ordering = list(hdf[group_index_label])
            
    return beamline(*elements, ordering=ordering)


def from_madx(filename, beta0, **kwargs):
    '''
    Load MAD-X lattice from file, using LatticeAdaptor module, 
    and construct a beamline object from the data.
    
    Parameters
    ----------
    filename: str
        The name of the MAD-X lattice to be loaded.
        
    beta0: float
        The realtivistic beta-factor (related to the energy of the beam). This is required later to
        build the Hamiltonians.
        
    **kwargs
        Optional arguments passed to 'to_beamline' routine.
        
    Returns
    -------
    beamline
        A beamline object representing the sequence of elements in the given lattice.
        
    Pandas dataframe
        A Pandas dataframe object, representing the loaded sequence of the lattice.
    '''
    la = LatticeAdaptor()
    la.load_from_file(filename, ftype='madx')
    raw_df = la.table
    
    # MAD-X specific labels
    position_label='at'
    length_label='L'
    bend_kx_label = 'K0'
    angle_label = 'ANGLE'
    component_labels = [bend_kx_label, 'K1', 'K2']
    madx_default_position = 0.5 # MAD-X tends to denote the position of the elements in the center

    if bend_kx_label not in raw_df.columns:
        # add kx
        angles = raw_df[angle_label].values
        lengths = raw_df[length_label].values 
        valid_indices = np.logical_and((~np.isnan(angles)), lengths > 0)
        kx = np.zeros(len(raw_df))
        kx[valid_indices] = angles[valid_indices]/lengths[valid_indices] # r*phi = L; kx = 1/r
        raw_df[bend_kx_label] = kx
    
    # drop elements with zero length and uneccesary columns;
    # N.B. E1 and E2 denote rotation angles of the pole-faces. If they are non-zero,
    # they are usually half the bend angle (in the rectangular case). We will ignore rectangular
    # bends for the time being and use s-bends here.
    columns_oi = [position_label, length_label] + component_labels
    # if they exist, add skew-values to the components; TODO: check & verify this numerically
    for j in range(1, 3):
        if f'K{j}S' in raw_df.columns:
            raw_df[f'K{j}'] = raw_df[f'K{j}'].values + raw_df[f'K{j}S']*1j
    raw_df = raw_df.loc[raw_df[length_label] > 0][columns_oi]
    
    # in MAD-X the respective values are the integrated field strengths. Therefore (TO BE CHECKED; TODO)
    #for j in range(1, 3):
    #    raw_df[f'K{j}'] = raw_df[f'K{j}'].values/raw_df[length_label]/fac
        
    # construct the sequence of Lie operators
    kwargs['position'] = kwargs.get('position', madx_default_position) 
    seq2 = to_beamline(raw_df, beta0=beta0, 
                      component_labels=component_labels, position_label=position_label,
                      length_label=length_label, **kwargs)
    return seq2, raw_df