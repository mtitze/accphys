from more_itertools import windowed
from collections import Counter

def greedy_nop(pattern, indices):
    '''
    For a given 'pattern' of indices which occurs in a larger list at certain 'indices', 
    find a solution with the largest number of non-overlapping patterns.
    
    The strategy is as follows:
    Let k be the first index of an optimal solution. If k is not the first occurence, then we can find
    a different optimal solution where k is replaced by the first index. The same goes for the last index.
    Therfore we can always start with the first and the last index in our search for an optimal
    solution and work our way into the interior from both sides simultaneously.
    This must always produce an optimal solution, although it may not be unique.
    '''
    if len(indices) == 0:
        return []
    size = len(pattern)
    b = indices[-1]
    indices_r = [k for k in indices[1:] if b - k >= size]
    solution = [indices[0]]
    for k in indices_r:
        if k - solution[-1] >= size:
            solution.append(k)
    return solution + [b]

def find_rno_patterns(sequence, abort=True, **kwargs):
    '''
    For a given sequence of integers of length n, find 
    repeating non-overlapping (rno) patterns
    between length 2 and n - 1.
    
    Parameters
    ----------
    sequence: list
        The list of integers to be examined.
        
    lengths: iterable, optional
        Search lengths. By default it will be range(len(sequence) - 1, 1, -1).
        
    abort: boolean, optional
        If true, abort the search if patterns (of a certain length) has been found.
    
    Returns
    -------
    dict
        A dictionary containing the patterns as keys and the indices where these
        patterns occur first as values (in form of lists).
    '''
    # taken & modified from https://stackoverflow.com/questions/61190161/is-there-a-faster-way-to-find-repeated-patterns-in-a-list
    n = len(sequence)
    result = {}
    for size in kwargs.get('lengths', range(n - 1, 1, -1)):
        result_size = {}
        windows = [tuple(pattern) for pattern in windowed(sequence, size)]
        counter = Counter(windows)
        for pattern, count in counter.items():
            if count <= 1:
                continue
            wsi = [j for j, w in enumerate(windows) if w == pattern] # wsi: pattern start indices
            if wsi[-1] - wsi[0] < size: # A non-overlapping solution is not possible
                continue
            # result[pattern] = wsi # would give all possible solutions of the pattern, including some which may overlap
            solution = greedy_nop(pattern, wsi)
            if len(solution) > 0:
                result_size[pattern] = solution
        if len(result_size) > 0:
            result[size] = result_size
        if abort and len(result) > 0:
            break
        # if a result has been found, abort the current loop.
    return result

def get_nested_patterns(rno_patterns, **kwargs):
    '''
    Obtain a list of nested repeating patterns, starting from the largest to the smallest.
    
    Remark:
    The routine is 'simple' in such a way that -- although it looks for large sequences --,
    there are no extensive checks to guarantee that the solution covers the
    largest number of indices. This may be optimized in a future version.
    '''
    heap = kwargs.get('heap', {}) # To store all chains which should be computed
    if len(rno_patterns) == 0:
        return heap
    # get a large chain with the most number of repetitions
    level = max(rno_patterns.keys())
    level_chains = rno_patterns[level]
    indmax = 2
    c_oi = next(iter(level_chains))
    ind_oi = level_chains[c_oi]
    for c, ind in level_chains.items():
        if len(ind) > indmax:
            c_oi = c
            ind_oi = ind
            
    if level not in heap.keys():
        heap[level] = {}
    if c_oi not in heap[level].keys():
        heap[level][c_oi] = ind_oi
        
    # get the constituents of that chain
    if level > 2:
        # there has to be one repeating element within heap[c_oi]
        rno_patterns_c = find_rno_patterns(c_oi)
        return get_nested_patterns(rno_patterns_c, heap=heap)
    else:
        return heap