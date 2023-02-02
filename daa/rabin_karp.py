# d is the number of characters in the input alphabet
d = 256

def search(pattern, main, q):
    """
    Rabin Karp algorithm for string matching. This algorithm is efficient
    for finding substring in a parent string.

    Parameters:
    pattern : str
              Substring that needs to be searched
    main : str
           Main string in which substring needs to be searched
    q : int
        q is a prime number, like 101

    Returns:
    i : int
        Index where substring is present
    
    """

    pat_len = len(pattern)
    main_len = len(main)
    i = 0
    j = 0
    p = 0    # hash value for pattern
    t = 0    # hash value for main
    h = 1
 
    # h is "pow(d, pat_len-1)% q"
    for i in range(pat_len-1):
        h = (h * d)% q

    # Calculate the hash value of pattern and first window of text
    for i in range(pat_len):
        p = (d * p + ord(pattern[i]))% q
        t = (d * t + ord(main[i]))% q
 
    # Slide the pattern over text 1 by 1
    for i in range(main_len-pat_len + 1):
        # Check the hash values of current window of text and pattern if the hash values match 
        # then only check for characters on by one
        if p == t:
            # Check for characters 1 by 1
            for j in range(pat_len):
                if main[i + j] != pattern[j]:
                    break
 
            j+= 1
            # if p == t and pattern[0...pat_len-1] = main[i, i + 1, ...i + pat_len-1]
            # if pattern found, return index
            if j == pat_len:
                return i
 
        # Calculate hash value for next window of text: 
        # Remove leading digit, add trailing digit
        if i < main_len-pat_len:
            t = (d*(t-ord(main[i])*h) + ord(main[i + pat_len]))% q
 
            # We might get negative values of t, converting it to positive
            if t < 0:
                t = t + q
