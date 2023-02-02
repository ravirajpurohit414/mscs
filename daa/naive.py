def search(pattern, main):
    """
    Naive method to search for a substring pattern in a main string
    
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
 
    # loop over the main string to look for the pattern
    for i in range(main_len - pat_len + 1):
        j = 0
        # For current index i, check for pattern match
        while(j < pat_len):
            if (main[i + j] != pattern[j]):
                break
            j += 1
 
        # return index where the pattern is found
        if (j == pat_len):
            return i
 