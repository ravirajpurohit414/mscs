## import libraries
import random
import string
import pandas as pd

## generate worst case dataset for string matching algorithm
## worst case is like - finding AAAB in a main string AAAAAAAAAAAAAAAAAB

## generate datasets of different sizes
# for i in [500,15000,25000]:
for i in [25000,100000,1000000]:
    data = {'main_str_data': [], 'sub_str_data': []}
    for j in range(i):
        mainstr_len = random.randint(100,130)
        substr_len = random.randint(3,5)

        main_str = ''
        letters = []
        
        while len(main_str) < mainstr_len:
            ## generate random number for repitition
            reps = random.randint(40,50)
            ## define reps according to the size left to be contained by main string
            if mainstr_len - len(main_str) < reps:
                reps = mainstr_len - len(main_str)
            
            ## generate a unique (not already in letters) random choice of alphabet
            random_letter = random.choices(string.ascii_uppercase, k = 1)[0]
            while random_letter in letters:
                random_letter = random.choices(string.ascii_uppercase, k = 1)[0]
            letters.append(random_letter)
            ## keep on appending until main string size is full
            main_str = main_str + (random_letter*reps)
        
        ## get the substring like AAAAAAB
        idx = random.randint(0, len(letters)-2)
        sub_str = letters[idx]*(substr_len-1) + letters[idx+1]

        data['main_str_data'].append(main_str)
        data['sub_str_data'].append(sub_str)

    pd.DataFrame(data).to_csv('data_worst_case'+str(i)+'.csv', index=False)
    # break