Reddit Post Classifier
----
Link to Jupyter Notebook - https://github.com/ravirajpurohit414/mscs/blob/main/dm/text_classifier_final_project/main.ipynb
Please refer to the interactive python notebook. Every step is well explained with comments and section are explained with markdowns.

Link to Medium Article - https://medium.com/@ravirajpurohit414/reddit-post-classifier-8ca186d17900

Steps to run/see
----
- Load the main.ipynb as a jupyter notebook
- Select Run all cells to run everything at once and see the outputs
NOTE - sometimes, text vectorization step takes some time.


Directory Structure -
----

- data/
- data/readme.txt - contains the links to download the datasets from kaggle
- data/Top_Posts_Comments.csv - the dataset containing comments for each posts
- data/Top_Posts.csv - the dataset containing reddit post information including the target variable

- main.ipynb - contains the code for the reddit post classifier problem with visualizations and printed outputs

- outputs/
- outputs/performance_comparison.png - model performances compared for each model
- outputs/comp_labels_with_comments.png - visualization of target variable separation with comments feature
- outputs/comp_labels_with_score.png - visualization of target variable separation with score feature
- outputs/comp_labels_with_upvoteratio.png - visualization of target variable separation with upvote ratio feature
- outputs/comp_labels_with_year.png - visualization of target variable separation with year feature

- docs/
- docs/reddit_post_classifier_blog.pdf - medium article blog post (linked above) saved as pdf

- requirements.txt - file containing the libraries used for the development of the script with versions

Development System Information -
----
matplotlib          3.5.1
nltk                3.7
numpy               1.21.5
pandas              1.4.2
session_info        1.0.0
sklearn             1.0.2
-----
Click to view modules imported as dependencies
-----
IPython             8.2.0
jupyter_client      6.1.12
jupyter_core        4.9.2
jupyterlab          3.3.2
notebook            6.4.8
-----
Python 3.9.12 (main, Apr  5 2022, 01:53:17) [Clang 12.0.0 ]
macOS-10.16-x86_64-i386-64bit