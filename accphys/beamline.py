from functools import reduce

from lieops import create_coords, combine

def identity(x):
    return x

def compose(f, g):
    return lambda x: f(g(x))

class beamline:
    
    def __init__(self, *sequence, **kwargs):
        '''
        Class to model an accelerator beamline.
        '''
        # consistency checks
        assert len(sequence) > 0
        assert all([hasattr(e, 'hamiltonian') for e in sequence])
        dim0 = sequence[0].hamiltonian.dim
        assert all([e.hamiltonian.dim == dim0 for e in sequence])
        
        self.dim = dim0
        self.sequence = sequence
        if 'lengths' in kwargs.keys():
            self.set_lengths(lengths=kwargs['lengths'])
        
    def set_lengths(self, lengths):
        n_elements = len(self.sequence)
        if len(lengths) > 0:
            assert len(lengths) == n_elements
        else:
            lengths = [1]*n_elements
        self.lengths = lengths
        
    def calcFlows(self, **kwargs):
        '''
        Calculate the sequence of Lie-operators exp(:f_n:) for each element f_n in self.sequence.
        
        Parameters
        ----------
        Parameters
        ----------
        lengths: list, optional
            Lengths of the elements. These lengths will modifying the respecive flows accordingly.
            If given, then exp(lengths[k]:x[k]:) will be computed. An additional t-argument will
            compute exp(t*lengths[k]:x[k]:).

        **kwargs
            Optional parameters passed to lieops.ops.lie.poly.flow
        '''
        if not hasattr(self, 'lengths') or 'lengths' in kwargs.keys():
            self.set_lengths(lengths=kwargs.get('lengths', []))
            
        n_elements = len(self.sequence)
        t_inp = kwargs.get('t', -1)
                    
        flows = []
        for k in range(n_elements):
            element = self.sequence[k]
            flows.append(element.hamiltonian.flow(t=t_inp*self.lengths[k], **kwargs))
        self.flows = flows
        
    def calcOneTurnMap(self, **kwargs):
        '''
        **kwargs are passed to create_coords routine (e.g. any possible max_power)
        '''
        assert hasattr(self, 'flows'), 'Need to call self.calcFlows first.'
        xiv = create_coords(self.dim, **kwargs)[:self.dim]
        composition = reduce(compose, self.flows, identity)
        self.oneTurnMap = composition(xiv)
        
    def __call__(self, point):
        assert hasattr(self, 'oneTurnMap'), 'Need to call self.calcOneTurnMap first.'
        return [c(point) for c in self.oneTurnMap]

    def magnus(self, power=1, **kwargs):
        '''
        Combine the individual Hamiltonians by means of Magnus series.
        
        Parameters
        ----------
        power: int, optional
            Power in terms of the free parameter (usually the longitudinal coordinate s),
            up to which the magnus series will be exact. 
        
        **kwargs
            Optional keyword arguments passed to lieops.ops.lie.combine routine
        '''
        
        hamiltonians = [e.hamiltonian for e in self.sequence]
        self._magnus_series, self._magnus_hamiltonian = combine(*hamiltonians, power=power, **kwargs)
        return sum(self._magnus_series.values())
        
        
        
        