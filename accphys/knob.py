from njet.common import check_zero

# Some comments:
# The knob class models polynomials in several variables and is very similar to njet's jetpoly class. 
# The difference is that the underlying polynomials are modeled by tuples instead of frozensets 
# (similar to the poly classes in lieops), and that a max_power parameter is used to control 
# the number of terms during multiplication (also similar to the poly class in lieops).

class knob:
    
    def __init__(self, dim, terms={}, max_power=float('inf')):
        self.dim = dim
        self.terms = terms
        self.max_power = max_power
        
    def __add__(self, other):
        add_terms = {k: v for k, v in self.terms.items()}
        if not isinstance(other, type(self)):
            zero = (0,)*self.dim
            new_value = add_terms.get(zero, 0) + other
            if new_value != 0:
                add_terms[zero] = new_value
        else:
            for k, v in other.terms.items():
                add_terms[k] = add_terms.get(k, 0) + v
        return self.__class__(terms={k: v for k, v in add_terms.items() if not check_zero(v)}, 
                              dim=self.dim, max_power=self.max_power)
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self.__class__(terms={k: -v for k, v in self.terms.items()}, 
                              dim=self.dim, max_power=self.max_power)
    
    def __sub__(self, other):
        return self + -other
    
    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        if not isinstance(other, type(self)):
            return self.__class__(terms={k: v*other for k, v in self.terms.items()}, 
                                  dim=self.dim, max_power=self.max_power)
        else:
            prod = {}
            max_power = min([self.max_power, other.max_power])
            for k1, v1 in self.terms.items():
                for k2, v2 in other.terms.items():
                    if any([k1[j] + k2[j] > max_power for j in range(self.dim)]):
                        continue
                    k = tuple(k1[j] + k2[j] for j in range(self.dim))
                    prod[k] = prod.get(k, 0) + v1*v2
            return self.__class__(terms={k: v for k, v in prod.items() if not check_zero(v)}, 
                                  dim=self.dim, max_power=self.max_power)   
            
    def __rmul__(self, other):
        return self*other
    
    def __truediv__(self, other):
        assert not check_zero(other)
        return self.__class__(terms={k: v/other for k, v in self.terms.items()}, 
                              dim=self.dim, max_power=self.max_power)

    def __pow__(self, other):
        assert type(other) == int
        assert other >= 0
        if other == 0:
            return self.__class__(terms={(0,)*self.dim: 1}, 
                                  dim=self.dim, max_power=self.max_power) # N.B. 0**0 := 1

        remainder = other%2
        half = self**(other//2)
        if remainder == 1:
            return self*half*half
        else:
            return half*half
        
    def __eq__(self, other):
        if not isinstance(self, type(other)):
            if len(self.terms) > 1:
                return False
            elif (0,)*self.dim in self.terms.keys():
                return self.terms[(0,)*self.dim] == other
            else:
                return False
        else:
            return self.terms == other.terms          
    
    def __str__(self):
        if len(self.terms) == 0:
            return '0'
        else:
            outstr = '['
            for key, value in self.terms.items():
                fac = ''
                for j in range(len(key)):
                    if key[j] == 0: # do not show z**0
                        continue
                    fac += f'*K{j}**{key[j]}'
                outstr += f'{value}]{fac} + \n ['
            return outstr[:-6]
    
    def _repr_html_(self):
        outstr = self.__str__().replace('\n', '<br>')
        return f'<samp>{outstr}</samp>'
    
def create_knobs(dim, **kwargs):
    '''
    Create a series of knobs to be used.
    '''
    return [knob(dim=dim, terms={tuple(0 if j != k else 1 for j in range(dim)): 1}, **kwargs) for k in range(dim)]
