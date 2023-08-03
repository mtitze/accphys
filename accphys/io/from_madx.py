from njet.common import factorials

import accphys
from cpymad.madx import Madx

def load_file(madx_file, sequence=None):
    '''
    Load a MAD-X file and retrun its cpymad lattice.
    
    Parameters
    ----------
    madx_file: str
        The location of the MAD-X lattice file.
        
    sequence: str, optional
        The name of the sequence to be returned. If nothing provided, it is
        assumed that the MAD-X file contains a single sequence.
        
    Returns
    -------
    cpymad.madx.sequence
    '''
    madx = Madx()
    madx.option(echo=False)
    madx.option(info=False)
    madx.option(warn=False)

    # load lattice
    madx.call(madx_file)
    seqd = madx.sequence
    if sequence == None:
        assert len(seqd) == 1, f'Sequence not specified: {seqd.keys()}'
        sequence = next(iter(madx.sequence))
    return madx.sequence[sequence]

def MadxElement2Elements(element, warn=True, disable_edges=False, **kwargs):
    '''
    Identify a MAD-X element and retreive the necessary information to generate one (or more) accphys element(s).

    Returns
    -------
    list
        A list of accphys.element objects, representing the given MAD-X element.

    list
        A list of the element(s) position(s) in the overall lattice.
    '''
    _skipped_types = ['hmonitor', 'monitor', 'vmonitor', 'instrument', 'marker', 'thinwire', 'wire', 'slmonitor', 'blmonitor',
                      'imonitor', 'changeref', 'translation', 'placeholder', 'collimator', 'changerefp0', 'sixmarker'] # some of these types can be obtained by madx = Madx(); madx.elements
    base_type = element._attr['base_type']
    if base_type in _skipped_types:
        return [], [element.position]

    # obtain generic MAD-X parameters 'length' and 'tilt' for the specific element
    madx_data = {k: v.value for k, v in element._data.items() if v.inform} # code for madx_data taken from __repr__ of cpymad elements
    length = kwargs.get('length', element._attr.get('length', madx_data.get('length', None)))
    if length == None:
        if warn:
            warnings.warn(f'Length of MAD-X element "{element.name}" not defined. Assuming zero.')
        length = 0
    tilt = kwargs.get('tilt', madx_data.get('tilt', 0))

    _madx_field_str_prefix = 'k' # prefix of how MAD-X denotes the components of the field strengths of multipoles

    # ----------------------------------------------
    # Translate the individual elements case by case
    # ----------------------------------------------
    
    # ------------- DRIFT -------------
    if base_type == 'drift':
        return [{'name': 'drift', 'length': length}], [element.position] # element.position gives the position at the entrance to the respective element (here: drift).
        
    # ------------- DIPEDGE -------------
    elif base_type == 'dipedge':
        raise NotImplementedError('dipedge parameters needs to be checked')
        
    # ------------- RBEND OR SBEND -------------
    elif base_type == 'rbend' or base_type == 'sbend':
        if 'angle' in madx_data.keys() and length != 0:
            angle = madx_data['angle']
            k0 = angle/length # r*phi = L; kx = 1/r
        else:
            k0 = madx_data.get('k0', 0) + 1j*madx_data.get('k0S', 0)

        component_indices = [int(k[1:]) for k in madx_data.keys() if k[0] == _madx_field_str_prefix and k[1:].isnumeric()]
            
        # initialize components
        N = max(component_indices)
        if len(component_indices) > 0:
            components = [0 for j in range(N + 1)]
            # facts = factorials(N)
        else:
            components = [0]
            
        # place values in the components
        for j in component_indices:
            if j == 0: # we deal with this case separately; component_indices may even not contain 0 anyways
                continue
            components[j] = madx_data.get(_madx_field_str_prefix + str(j), 0)# /facts[j] # TODO: Check factorials here
        components[0] = k0
                
        out_bend = {'length': length, 'tilt': tilt}
        if len([c for c in components if c != 0]) > 1: # if this condition is true, we have a combined-function magnet
            out_bend['name'] = 'cfm'
            out_bend['components'] = tuple(components) # conversion to tuple as a temp step to later get the unique elements, which requires hashable objects
        else:
            out_bend['name'] = 'dipole'
            out_bend['args'] = (k0,)

        if (base_type == 'rbend' or 'e1' in madx_data.keys() or 'e2' in madx_data.keys()) and k0 != 0 and not disable_edges:
            # also add pole-face rotation matrices:
            rho = 1/k0
            angle = length*k0
            e1 = madx_data.get('e1', angle//2)
            e2 = madx_data.get('e2', angle//2)
            # add two (thin) pole-face rotation elements in front and behind
            return [{'name': 'polefaceRM', 'rho': rho, 'phi': e1}, out_bend, {'name': 'polefaceRM', 'rho': rho, 'phi': e2}], [element.position, element.position, element.position + length]
        else:
            return [out_bend], [element.position]

    # ------------- QUADRUPOLE -------------
    elif base_type == 'quadrupole':
        return [{'name': 'quadrupole', 'args': (madx_data['k1'],), 'tilt': tilt, 'length': length}], [element.position]

    # ------------- SEXTUPOLE -------------
    elif base_type == 'sextupole':
        return [{'name': 'sextupole', 'args': (madx_data['k2']/2,), 'tilt': tilt, 'length': length}], [element.position]

    # ------------- OCTUPOLE -------------
    elif base_type == 'octupole':
        return [{'name': 'octupole', 'args': (madx_data['k3']/6,), 'tilt': tilt, 'length': length}], [element.position]
        
    else:
        if warn:
            warnings.warn(f'Element of type {base_type} not implemented.')
        return [], [element.position]

def to_beamline(madx_file, **kwargs):
    '''
    Load a MAD-X lattice file and convert it to a beamline object.
    '''
    lat = load_file(madx_file)
    if hasattr(lat, 'beam'):
        _ = kwargs.setdefault('beta0', lat.beam.beta)
    return accphys.io.convert.Sequence2Beamline([e for e in lat.elements], **kwargs)
