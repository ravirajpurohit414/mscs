## import libraries
import random
import string
import pandas as pd

## this is a script to generate synthetic data for string matching
## we are creating a random sized main string and slicing a substring from a random location from within

## generate datasets of different sizes
for i in [500,15000,25000]:
    data = {'main_str_data': [], 'sub_str_data': []}
    for j in range(i):
        ## define a random size for the main string
        mainstr_len = random.randint(100,150)
        ## define a random size for the substring
        substr_len = random.randint(3,5)
        ## define an index for choosing the substring from the main string
        idx = random.randrange(0, mainstr_len - substr_len + 1)

        ## generate the main string
        main_str = ''.join(random.choices(string.ascii_uppercase, k = mainstr_len))
        ## slice the substring
        sub_str = main_str[idx : idx+substr_len]

        data['main_str_data'].append(main_str)
        data['sub_str_data'].append(sub_str)

    pd.DataFrame(data).to_csv('data'+str(i)+'.csv', index=False)
    # break