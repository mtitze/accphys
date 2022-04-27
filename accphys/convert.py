from .beamline import beamline
from .elements import cfm
from tqdm.auto import tqdm

def prepare(hdf, position_label='s', length_label='L', position=0, tol=1e-6, **kwargs):
    '''
    Prepare a given data frame so that there will be no zero-spaces in between two elements.
    
    Parameters
    ----------
    hdf: pd.Dataframe
        Pandas dataframe object defining the beam sequence. The dataframe must contain the
        position_label and length_label columns.
    
    position: float, optional
        Defines the alignment of the elements relative to their actual position.
        Take a value between 0 and 1, where 0 means the position is given relative to the element start,
        while 1 means that they are taken with respect to the element end.
    
    tol: float, optional
        Tolerance below which we consider the end/start of two adjacent elements to agree with each other.
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
            raise RuntimeError(f"Element {k} at position {pos} with length {length} appears to overlap its successor at position {next_pos}."\
                               + " Check lattice input and/or 'position' and 'tol' arguments.")
            
        #sequence.append(cfm(components=[current_row[label] for label in component_labels], length=length, **kwargs))

        if pos + length + tol < next_pos:
            # here we have to insert additional drift space in between.
            empty_space = next_pos - pos - length
            
            new_row = [0]*len(hdf.columns)
            new_row[position_label_index] = pos + length
            new_row[length_label_index] = empty_space
            hdf.loc[k + 0.5] = new_row # insert a new row in between k and k + 1
            
    return hdf.sort_index().reset_index(drop=True)


def toBeamline(hdf, beta0, component_labels=['kx0', 'kx1', 'kx2', 'kx3'], position_label='s', length_label='L', **kwargs):
    '''
    Construct a beamline from a given lattice.
        
    **kwargs
        Keyworded arguments passed to elements. 
    '''
    # Preparation; ensure that no empty space exists between elements (they will be filled with drifts if necesary):
    hdf = prepare(hdf, position_label=position_label, length_label=length_label, **kwargs)
            
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