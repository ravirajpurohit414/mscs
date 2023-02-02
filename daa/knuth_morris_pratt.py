def search(pattern, main):
    """
    Knuth Morris Pratt algorithm for string matching. This algorithm is efficient
    for finding substring in a parent string.

    Parameters:
    pattern : str
              Substring that needs to be searched
    main : str
           Main string in which substring needs to be searched

    Returns:
    i : int
        Index where substring is present
    """

    pat_len = len(pattern)
    main_len = len(main)
 
    # create lps list that will hold the longest preffix and suffix for pattern
    lps = [0]*pat_len
    # index for pattern
    j = 0 
 
    # Preprocess the pattern (calculate lps list)
    compute_lps_list(pattern, pat_len, lps)
 
    i = 0 # index for main list
    while i < main_len:
        if pattern[j] == main[i]:
            i += 1
            j += 1
 
        if j == pat_len:
            # return index when pattern found
            return i-j
            # j = lps[j-1]
 
        # mismatch after j matches
        elif i < main_len and pattern[j] != main[i]:
            # Do not match lps[0..lps[j-1]] characters they'll match anyway
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
 
def compute_lps_list(pattern, pat_len, lps):
    len = 0 # length of the previous longest prefix and suffix
 
    lps[0] # lps[0] is always 0
    i = 1
 
    # the loop calculates lps[i] for i = 1 to pat_len-1
    while i < pat_len:
        if pattern[i]== pattern[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # Consider a tricky example.
            # AAACAAAA and i = 7. The idea is similar to search step.
            if len != 0:
                len = lps[len-1]
                # don't increment i
            else:
                lps[i] = 0
                i += 1