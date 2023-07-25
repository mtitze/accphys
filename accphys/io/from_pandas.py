from accphys import beamline
from accphys.elements import cfm

from tqdm import tqdm

def _fill_gaps(hdf, position_label='s', length_label='L', position=0, tol=1e-6, **kwargs):
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
    


def from_pandas(hdf, component_labels, component_indices, position_label='s', length_label='L', **kwargs):
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
    hdf = _fill_gaps(hdf, position_label=position_label, length_label=length_label, **kwargs)
            
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

        element = cfm(components=components, length=length, **kwargs)
        # special case: check for rbends etc. (TODO: improve this code, as it depends on the labels E1 and E2)
        # we add sector rotation matrices if required.
        #e1, e2 = row.get('E1', 0), row.get('E2', 0)
        #if e1 != 0:
        #    elements.append(polefaceRM(rho=element.rhox, phi=e1, length=0))

        elements.append(element)

        #if e2 != 0:
        #    elements.append(polefaceRM(rho=element.rhox, phi=e2, length=0))
        group_index += 1
        
    # set the ordering
    ordering = list(hdf[group_index_label])  # ONGOING: need to separate the edges in advance before determining the unique groups of elements...
            
    return beamline(*elements, ordering=ordering)


