NO_OF_CHARS = 256
 
def bad_char_heuristic(string, size):
    '''
    The preprocessing function for
    Boyer Moore's bad character heuristic
    
    Parameters:
    string : str
             String to find the bad character in
    size : int
           Size of string

    Returns:
    badChar : int
        Bad character
    '''
 
    # Initialize all occurrence as -1
    badChar = [-1]*NO_OF_CHARS
 
    # Fill the actual value of last occurrence
    for i in range(size):
        badChar[ord(string[i])] = i
 
    # return initialized list
    return badChar
 
def search(main, pattern):
    '''
    A pattern searching function that uses Bad Character
    Heuristic of Boyer Moore Algorithm
    
    Parameters:
    pattern : str
              Substring that needs to be searched
    main : str
           Main string in which substring needs to be searched

    Returns:
    i : int
        Index where substring is present    

    '''
    pat_len = len(pattern)
    main_len = len(main)
 
    # create the bad character list by calling the preprocessing function 
    # bad_char_heuristic() for given pattern
    badChar = bad_char_heuristic(pattern, pat_len)
 
    # s is shift of the pattern with respect to text
    s = 0
    while(s <= main_len-pat_len):
        j = pat_len-1
 
        # Keep reducing index j of pattern while characters of 
        # pattern and text are matching at this shift s
        while j>=0 and pattern[j] == main[s+j]:
            j -= 1
 
        # If the pattern is present at current shift, 
        # then index j will become -1 after the above loop
        if j<0:
            # print("Pattern occur at shift = {}".format(s))
            return s
 
            # Adjust the pattern so that the following character in 
            # the text lines up with the last time it appears there
            # For the case of pattern occuring at the end of text
            # the condition s+pat_len < main_len is necessary
            s += (pat_len-badChar[ord(main[s+pat_len])] if s+pat_len<main_len else 1)
        else:
            # Adjust the pattern so that the bad character in the 
            # text lines up with the final instance of it there.
            #
            # The max function makes sure a +ve shift. 
            # There might be a -ve shift if last occurrence of the bad character 
            # in pattern is on the right side of the current character
            s += max(1, j-badChar[ord(main[s+j])])
 