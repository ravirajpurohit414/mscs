Project - String Matching Algorithms performance comparison

To run the code, please follow the below mentioned flow -
1. The Algorithms are coded in python. The files are -
naive.py
knuth_morris_pratt.py
rabin_karp.py
boyer_moore.py

2. The idea is to compare performances of the Algorithms. For this, we need data.
The data is generated synthetically.
There are two experiments done - 

Experiment 1 - Normal case dataset
Run generate_data.py file to generate normal case dataset
for example - 
Main string  - ABCDEFGHIJKLMNOP
Pattern string - JKL

We generated 500, 15000 and 25000 samples as datasets and ran all the algorithms 10 times on each dataset 
to get a good idea of average running time.

Experiment 2 - Worst case dataset
Run generate_data_worst_case.py to generate worst case dataset
In the experiment 1, we found out that the naive algorithm was performing much better than the 
best considered (theoretically) Knuth-Morris-Pratt (KMP) search algorithm.
So, upon analysis, we found out that the KMP algorithm provides advantage when there is repition in data.
For example, when data is like -
Main string - AAAAAAAAAAAAAAAAAB
Pattern string - AAB

So, we generated such random data of 25000, 100000 and 1000000 samples as dataset. We increased the number of samples
to get a significant number while comparing time of execution.

After running data generation scripts, we get .csv files as our datasets to work with.

3. Performance Comparison
Look at/Run the performance_comparison.ipynb file to view the performance comparison.
This is an interactive python notebook, hence the output will be available right away.

In this script, we are reading the experiment 1 and experiment 2 files respectively to compare their performacnes


Please look at the report pdf file for more details.
Individual Contribution is mentioned at the end of the report.
Thank you.


Authors - 
Ravi Rajpurohit (1002079916)
Vedanti Ambulkar (1001829121)