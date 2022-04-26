from tqdm.auto import tqdm # progress bar

from .beamline import beamline
from .elements import cfm

def toBeamline(hdf, component_labels=['kx0', 'kx1', 'kx2', 'kx3'], position_label='s', length_label='L', 
               position=0, tol=1e-6, **kwargs):
    '''
    Construct a beamline from a given lattice.
    
    Parameters
    ----------
    
    position: float, optional
        Defines the alignment of the elements relative to their actual position.
        Take a value between 0 and 1, where 0 means the position is given relative to the element start,
        while 1 means that they are taken with respect to the element end.
    
    tol: float, optional
        Tolerance below which we consider the end/start of two adjacent elements to agree with each other.
        
    **kwargs
        Keyworded arguments passed to the elements. 
    '''
    toStartPos = lambda z, length: z - length*position # transform position to start of element.

    sequence = []
    for k in tqdm(range(len(hdf))):
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
            
        sequence.append(cfm(components=[current_row[label] for label in component_labels], length=length, **kwargs))

        if pos + length + tol < next_pos:
            # here we have to insert additional drift space in between.
            empty_space = next_pos - pos - length
            sequence.append(drift(length=empty_space, **kwargs))
            
    return beamline(*sequence)