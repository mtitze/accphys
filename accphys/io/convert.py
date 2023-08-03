import cpymad

import accphys
from accphys.elements import drift, dipole, polefaceRM, cfm, quadrupole, sextupole, octupole
from accphys.io.from_madx import MadxElement2Elements

def Sequence2Elements(latticeElements, tol=0, **kwargs):
    '''
    Take a lattice object and extract the necessary information out of it,
    in order to create accphys elements.
    
    Parameters
    ----------
    lattice: list
        A list of elements given by one of the following codes:
        1) cpymad
    
    tol: float, optional
        An optional parameter to detect overlaps and gaps in between the sequence. In case of gaps,
        a drift will be added. In case of overlaps, an error will be issued.
        
    **kwargs
        Optional keyworded arguments passed to the underlying converter.
        
    Returns
    -------
    list
        A list of dictionaries, each containing the essential information to create an accphys element. Hereby
        the order of the list corresponds to the sequence (lattice) in question.
    '''
    assert tol >= 0, 'Tolerance can not be negative.'
    
    # probe input & select converter
    if all([isinstance(e, cpymad.madx.Element) for e in latticeElements]):
        converter = MadxElement2Elements
    else:
        raise NotImplementedError('Input list contains unsupported objects.')

    prev_position = 0
    chain = []
    for e in latticeElements:
        elements, positions = converter(e, **kwargs)
        if len(elements) == 0:
            continue

        diff = positions[0] - prev_position
        assert diff > -tol, f'Elements appear to overlap each other, using a tolerance (tol) of {tol}.'
        if diff > tol:
            # add a drift element to fill empty gap
            chain.append([{'name': 'drift', 'length': diff}])
        chain.append(elements)
        prev_position = positions[-1] + elements[-1].get('length', 0)

    return [ee for e in chain for ee in e] # flatten

def combine_adjacent_elements(elist, info=True):
    '''
    Combine adjecent elements in an overal list, if they contain the same information besides
    of their lengths.
    
    Parameters
    ----------
    elist: list
        A list of dictionaries, each containing information for the construction of an accphys element.

    info: boolean, optional
        Display some information.
        
    Returns
    -------
    list
        A new list in which suitable neighbouring elements have been combined.
    '''
    e0 = {}
    e0_length = 0
    indices_to_be_removed = [] # store information of those indices in the original list which must be removed.
    j = 0
    n_combined = 0 # count the number of elements to be combined to display this number to the user later.
    for e in elist:
        ee = {k: v for k, v in e.items() if k != 'length'}
        if ee == e0 and j > 0:
            e['length'] = e.get('length', 0) + e0_length
            indices_to_be_removed.append(j - 1)
            n_combined += 1
        e0_length = e.get('length', 0)
        e0 = ee
        j += 1
    if info and n_combined > 0:
        print (f'{n_combined} elements have been combined.')
    return [elist[j] for j in range(len(elist)) if j not in indices_to_be_removed]

def _createElement(name, **parameters):
    '''
    Create a beamline element out of a given set of parameters.
    '''
    args = parameters.pop('args', None)
    if name == 'drift':
        return drift(**parameters)
    elif name == 'cfm':
        return cfm(**parameters)
    elif name == 'dipole':
        return dipole(*args, **parameters)
    elif name == 'quadrupole':
        return quadrupole(*args, **parameters)
    elif name == 'sextupole':
        return sextupole(*args, **parameters)
    elif name == 'octupole':
        return octupole(*args, **parameters)
    elif name == 'polefaceRM':
        return polefaceRM(**parameters)
    else:
        raise NotImplementedError(f"Element of name '{name}' not understood.")

def Sequence2Beamline(sequence, tol_lat=1e-6, info=True, warn=True, combine=True, **kwargs):
    '''
    Convert a given list of elements into an accphys beamline.
    
    Parameters
    ----------
    tol_lat: float, optional
        Tolerance given to combined_adjacent_elements routine.

    info: boolean, optional
        Display some information.

    warn: boolean, optional
        Toggle on/off warnings.

    combine: boolean, optional
        Combine adjacent elements with identical parameters.
        
    **kwargs
        Optional keyworded arguments passed to the construction of the accphys elements. In particular it
        may be required to provide a beta0 (or energy) parameter.

    Returns
    -------
    beamline
        An accphys beamline object.
    '''
    sequence_with_ordering = Sequence2Elements(sequence, tol=tol_lat, warn=warn, **kwargs)
    if combine:
        sequence_with_ordering = combine_adjacent_elements(sequence_with_ordering, info=info)
    unique_elements = [dict(s) for s in set(frozenset(d.items()) for d in sequence_with_ordering)] # trick discussed in https://stackoverflow.com/questions/11092511/list-of-unique-dictionaries
    ordering = [unique_elements.index(e) for e in sequence_with_ordering]
    for u in unique_elements: # add user-specific parameters for the construction of elements
        u.update(**kwargs)
    ue = [_createElement(warn=warn, **ee) for ee in unique_elements]
    return accphys.beamline(*ue, ordering=ordering)