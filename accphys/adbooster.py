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

def find_rno_patterns(sequence, abort=True, min_occurences=2, **kwargs):
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
        
    size: int
        The length of the pattern at which the algorithm stopped.
    '''
    # taken & modified from https://stackoverflow.com/questions/61190161/is-there-a-faster-way-to-find-repeated-patterns-in-a-list
    n = len(sequence)
    result = {}
    sizes = kwargs.get('lengths', range(n - 1, 1, -1))
    size = 0
    for size in sizes:
        result_size = {}
        windows = [tuple(pattern) for pattern in windowed(sequence, size)]
        counter = Counter(windows)
        for pattern, count in counter.items():
            if count < min_occurences:
                continue
            wsi = [j for j, w in enumerate(windows) if w == pattern] # wsi: pattern start indices
            if len(wsi) > 1 and wsi[-1] - wsi[0] < size: # A non-overlapping solution is not possible
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
    return result, size


def _get_nested_patterns(rno_patterns, **kwargs):
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
        rno_patterns_c, _ = find_rno_patterns(c_oi)
        return _get_nested_patterns(rno_patterns_c, heap=heap)
    else:
        return heap
    
    
def _first_ref_pattern_positions(npat):
    '''
    Get the positions of the patterns in the reference sequence, assuming that
    the positions of the largest pattern is related to the reference sequence.
    '''
    lengths = sorted(npat.keys(), reverse=True)
    if len(lengths) == 0:
        return [], []
    _pat_pos = lambda key: list(npat[key].items())[0] # for a given level (key), obtain the pattern and the position(s)
    abs_position = _pat_pos(lengths[0])[1][0] # the position of the first occurence of the largest pattern
    positions = [abs_position]
    for length in lengths[1:]:
        first_relative_position = _pat_pos(length)[1][0]
        positions.append(first_relative_position + positions[-1])
    return positions, lengths
    
    
def _spaces_per_level(npat):
    levels = sorted(npat.keys())
    _pat_pos = lambda key: list(npat[key].items())[0] # for a given level (key), obtain the pattern and the position(s)
    spaces_in_level = []
    for level in levels:
        pattern, pos = _pat_pos(level)
        n_pos = len(pos) # number of occurences of pattern in next level
        space = len(pattern)*n_pos # space pattern copies occupy in next level
        for e in spaces_in_level:
            if len(e) > 0:
                sp, po = e[-1]
                e.append((sp*n_pos, po*n_pos))
        spaces_in_level.append([(space, n_pos)])
    # the k-th entry ek of 'spaces_in_level' is a list of integers. It is related to level[k].
    # ek[j] consists of two numbers:
    # 1) The space occupied of level[k] in level[k + j + 1].
    # 2) The total number of occurences of level[k] in level[k + j + 1]
    return spaces_in_level

    
def _merge_descent(cyc):
    '''
    Merge a given chain in such a way that we merge through a sequence of large sub-patterns.
    '''
    npat = _get_nested_patterns(find_rno_patterns(cyc.ordering)[0])    
    _pat_pos = lambda key: list(npat[key].items())[0]
    abs_pos, levels = _first_ref_pattern_positions(npat)
    levels = levels[::-1]
    abs_pos = abs_pos[::-1]

    spl = _spaces_per_level(npat)
    
    cycr = cyc
    level_sizes = {k: len(_pat_pos(levels[k])[0]) for k in range(len(levels))}
    for k in range(len(levels) - 1):
        level = levels[k]
        abs_pattern, positions = _pat_pos(level) # abs_pattern: current pattern, but relative to the original ordering
        n_positions = len(positions) # the number of the current pattern occuring within the next pattern.
        size = level_sizes[k]
        pattern = tuple(cycr.ordering[abs_pos[k]: abs_pos[k] + size])
        #print (size, pattern, positions[0], n_positions, abs_pos[k])
        cycr = cycr.merge(pattern=pattern)
        
        # modify all sizes accordingly
        space_k = spl[k]
        for l in range(len(space_k) - 1):
            space, n_occurences = space_k[l] # occupied space of the pattern in level k + l + 1 and occurences
            level_sizes[k + l + 1] = level_sizes[k + l + 1] - n_occurences*size + n_occurences # "+ n_positions", because 'n_positions' new elements are inserted
    return cycr
    
def pattern_cycle(*point, dchain, **kwargs):
    cyc = dchain.cycle(*point, outf=None, **kwargs)
    cycr = _merge_descent(cyc)
    return cycr.compose()