#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Project 1
# ---

# In[1]:


## import required libraries
import numpy as np
import matplotlib.pyplot as plt

## global variables
k = 9
gamma = 0.148


# In[2]:


def format_train_test_data(dataset):
    """
    A function to properly structure the dataset given for 
    machine learning and split into features and labels.
    
    Parameters
    ----------
    dataset - list of str
    txt file loaded as working data
    
    Returns
    -------
    data - numpy array of float
    features to use for machine learning
    
    labels - numpy array of float
    target label to train and predict
    """
    
    data = []
    labels = []

    for i in [i[:-2].split(',') for i in dataset]:
        data.append(float(i[0]))
        labels.append(float(i[1]))
    
    return np.array(data), np.array(labels)


# In[3]:


## load data
with open('train.txt') as f:
    train_dataset = f.readlines()
    f.close()
    
with open('test.txt') as f:
    test_dataset = f.readlines()
    f.close()


# In[4]:


## split the data samples and target labels
train_data, train_labels = format_train_test_data(train_dataset)
test_data, test_labels = format_train_test_data(test_dataset)


# ## Linear Regression
# ---

# In[5]:


train_data


# In[6]:


train_labels


# In[7]:


test_data


# In[8]:


test_labels


# In[9]:


def gen_feature_matrix(data, k, d):
    """
    This function computes the basis functions for a given 
    set of x-values, frequency increment k, and function depth d.
    
    Parameters
    ----------
    data - numpy array of float
    
    k - int
    global value defined at the top
    
    d - int
    variable ranging from 0 to 6
    

    Returns
    -------
    a numpy array of shape (N, 2*d+2), where N is the number of x-values.
    """
    
    sin_cos = np.empty((len(data), 2*d))
    for i in range(d):
        sin_cos[:, 2*i] = ((np.sin((i+1)*k*data))**(i*k))*np.cos(data)
        sin_cos[:, 2*i+1] = ((np.cos((i+1)*k*data))**(i*k))*np.sin(data)
    
    return np.column_stack((np.ones_like(data), data, sin_cos, np.sin(k*data), np.cos(k*data)))


# In[10]:


### ------------------------------------------------------------------------------- ###
### ------------------------------ PROBLEM 1, PART A ------------------------------ ###
### ------------------------------------------------------------------------------- ###
def linear_regression(x_train, y_train, x_test, y_test, k, d):
    """
    Fits a linear regression model to the training data and evaluates its performance on the testing data.
    
    Parameters
    ----------
    x_train - numpy array of float
    training data features
    
    y_train - numpy array of float
    training labels
    
    x_test - numpy array of float
    testing data features
    
    y_test - numpy array of float
    testing labels
    
    k - int
    global value defined at the top
    
    d - int
    variable ranging from 0 to 6
    
    Returns
    -------
    
    beta -
    the fitted parameter vector
    
    y_pred_train - 
    
    y_pred_test -
    
    mse - 
    the mean squared error (MSE) on the testing data
    
    
    """
    
    X_train = gen_feature_matrix(x_train, k, d)
    X_test = gen_feature_matrix(x_test, k, d)
    
    # compute the MPP of the design matrix
    X_pinv = np.linalg.pinv(X_train)    
    # fit the model to the training data
    beta = np.dot(X_pinv, y_train)
    
    # evaluate the model on the train data
    y_pred_train = np.dot(X_train, beta)
    # evaluate the model on the train data
    y_pred_test = np.dot(X_test, beta)
    
    mse = np.mean((y_test - y_pred_test)**2)
    
    return beta, y_pred_train, y_pred_test, mse


# ## Linear Regression Execution

# In[11]:


## Performance analysis on train_data

## -------- RUN ALL THE ABOVE CELLS BEFORE RUNNING THIS --------

def problem1(train_data, train_labels, test_data, test_labels, k):
    """
    Execution of linear regression problem 1 for project 1
    """
    
    train_sorted_ind = np.argsort(train_data)
    test_sorted_ind = np.argsort(test_data)

    mse_list = []

    ### ------------------------------------------------------------------------------- ###
    ### --------------------------- PROBLEM 1, PART B and C --------------------------- ###
    ### ------------------------------------------------------------------------------- ###

    for d in range(7):
        beta, y_pred_train, y_pred_test, mse = linear_regression(train_data, 
                                                                 train_labels, 
                                                                 test_data, 
                                                                 test_labels, 
                                                                 k, 
                                                                 d)

        mse_list.append(mse)

        ## plot figure for each iteration
        plt.figure(figsize=(16,4))
        plt.subplot(121)
        title = "TRAIN-DATA\nLinear Regressor Function for d = "+str(d)
        plt.title(title)
        plt.scatter(train_data[train_sorted_ind], train_labels[train_sorted_ind])
        plt.plot(train_data[train_sorted_ind], y_pred_train[train_sorted_ind])
        labels = ["data","lin Regressor"]
        plt.legend(labels)

        plt.subplot(122)
        title = "TEST-DATA\nLinear Regressor Function for d = "+str(d)+", and MSE = "+str(round(mse,4))
        plt.title(title)
        plt.scatter(test_data[test_sorted_ind], train_labels[test_sorted_ind])
        plt.plot(test_data[test_sorted_ind], y_pred_test[test_sorted_ind])
        labels = ["data","lin Regressor"]
        plt.legend(labels)

        plt.show()
    
    return mse_list


mse_list = problem1(train_data, train_labels, test_data, test_labels, k)


# In[12]:


mse_list


# ### Linear Regression Conclusion
# ---
# 
# In this program, I designed a linear regressor and tested its performance using test data. I am using Mean Squared Error (MSE) to determine the performance of the function.
# 
# In the above cell, we can see the output of variable mse_list
# - The index signifies the depth value (d) 
# - The value signifies the MSE for that depth (d)
# 
# #### Here are my <u>observations</u> about the regression function with variable function depths
# - The data itself is pretty scattered
# - For d = 1, the function is catching the trend of data points and is producing second lowest MSE for test data
# - For d = 4, the MSE for test data is lowest
# - But as discussed in class, I would choose the **regressor with function depth 1 to be the best model**, because it gives MSE close to minimum while capturing the natural trend of data
# - I believe that as I am increasing the function depth, the model is overfitting. We can see the same thing using MSE as it starts to increase a little as d increases, precisely after d = 1.

# # ----------------------------------------------------------------------------------

# ## Locally Weighted Linear Regression
# ---

# In[13]:


def calculate_weights(X, x_query, gamma):
    """
    calculate local weights for Linear Regression
    Parameters
    ----------
    X - numpy array of float
    train data
    
    x_query - numpy array of float
    local data
    
    gamma - float
    constant value given for the regression function
    
    Returns
    -------
    calculated local weights with respect to x_query
    """
    return np.exp(-(X - x_query) ** 2 / (2 * gamma ** 2))

### ------------------------------------------------------------------------------- ###
### ------------------------------ PROBLEM 2, PART A ------------------------------ ###
### ------------------------------------------------------------------------------- ###
def locally_weighted_linear_regression(X_train, y_train, x_query, gamma):
    """
    function for linear regression optimized by local weights
    
    Returns
    -------
    y_pred - float
    prediction using the model function
    """
    
    weights = calculate_weights(X_train[:,1], x_query, gamma)
    W = np.diag(weights.flatten())
    temp = np.dot(X_train.T,W)
    theta = np.linalg.inv(X_train.T @ W @ X_train) @ X_train.T @ W @ y_train
    y_pred = (np.array([1, x_query]).T) @ theta

    return y_pred


# In[14]:


## prepare data for only raw feature and a constant feature
def prepare_train_test_data_lwlr(data):
    """
    to format the train-test data for local weighted linear regression problem
    
    Parameters
    ----------
    data - numpy array of float
    
    Returns
    -------
    X - numpy array of float
    two dimensional array with constant feature
    """
    
    X = np.zeros((len(data),2))
    X[:,0] = 1
    X[:,1] = data
    
    return X


# In[15]:


## -------- RUN THE ABOVE CELLS UPTO RELEVANT HEADING BEFORE RUNNING THIS --------
def problem2(train_data, train_labels, test_data, test_labels):
    """
    Execution of Local Weighted Linear Regression problem 2 for project 1
    """
    X_train = prepare_train_test_data_lwlr(train_data)
    y_train = train_labels

    X_test = prepare_train_test_data_lwlr(test_data)
    y_test = test_labels

    ## execute the linear regression for local weights
    y_pred_train = []
    y_pred_test = []

    for train_row in train_data:
        # run locally weighted linear regression for train data
        y_pred_temp = locally_weighted_linear_regression(X_train, y_train, train_row, gamma)
        y_pred_train.append(y_pred_temp)

    for test_row in test_data:
        try:
            # run locally weighted linear regression for test data
            y_pred_temp = locally_weighted_linear_regression(X_train, y_train, test_row, gamma)
            y_pred_test.append(y_pred_temp)
        except:
            print('\nSKIPPING LAST INSTANCE BECAUSE\nMATRIX INVERSE NOT POSSIBLE FOR INSTANCE x_test = {}'.format(test_row))
            
    mse = np.mean((y_test[-1] - y_pred_test)**2)
            
    ### ------------------------------------------------------------------------------- ###
    ### --------------------------- PROBLEM 2, PART B and C --------------------------- ###
    ### ------------------------------------------------------------------------------- ###
    
    train_sorted_ind = np.argsort(train_data)
    ## skipping last test instance as matrix inverse not possible, hence can not plot for that
    test_sorted_ind = np.argsort(test_data[:-1])
    
    ## plot figure for performance analysis
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    title = "TRAIN-DATA\nLocal Weighted Lin Regressor Fn"
    plt.title(title)
    plt.scatter(train_data[train_sorted_ind], train_labels[train_sorted_ind])
    plt.plot(train_data[train_sorted_ind], np.array(y_pred_train)[train_sorted_ind])
    labels = ["data","local-wt lin-Reg"]
    plt.legend(labels)

    plt.subplot(122)
    title = "TEST-DATA\nLocal Weighted Lin Regressor Fn, and MSE = "+str(round(mse,4))
    plt.title(title)
    plt.scatter(test_data[test_sorted_ind], train_labels[test_sorted_ind])
    plt.plot(test_data[test_sorted_ind], np.array(y_pred_test)[test_sorted_ind])
    labels = ["data","local-wt lin-Reg"]
    plt.legend(labels)

    plt.show()
    
    """
    Part C explanation
    ------------------
    When I compare the performance of linear regression model in problem 1 
    and locally weighted linear regression in problem 2 on test data,
    - I found that the locally weighted regressor performs significantly better.
    - The evidence is provided by the Mean Squared Error. 
    - In case of problem 1, the test data had the best MSE of 0.16, 
    - while in problem 2, the MSE for test data is 0.058 using local weights.
    
    Hence Locally Weighted Linear Regression is better performing in this case.
    """
    
    
    ### ------------------------------------------------------------------------------- ###
    ### ------------------------------ PROBLEM 2, PART D ------------------------------ ###
    ### ------------------------------------------------------------------------------- ###
    
    ## execute the linear regression for local weights
    y_pred_train = []
    y_pred_test = []

    for train_row in train_data[:20]:
        # run locally weighted linear regression for train data
        y_pred_temp = locally_weighted_linear_regression(X_train[:20], y_train[:20], train_row, gamma)
        y_pred_train.append(y_pred_temp)

    for test_row in test_data:
        try:
            # run locally weighted linear regression for test data
            y_pred_temp = locally_weighted_linear_regression(X_train[:20], y_train[:20], test_row, gamma)
            y_pred_test.append(y_pred_temp)
        except:
            print('\nSKIPPING LAST INSTANCE BECAUSE\nMATRIX INVERSE NOT POSSIBLE FOR INSTANCE x_test = {}'.format(test_row))
    
    mse = np.mean((y_test[-1] - y_pred_test)**2)
    
    train_sorted_ind = np.argsort(train_data[:20])
    ## skipping last test instance as matrix inverse not possible, hence can not plot for that
    test_sorted_ind = np.argsort(test_data[:-1])
    
    print("------------ PART D ------------")
    ## plot figure for performance analysis
    plt.figure(figsize=(16,4))
    plt.subplot(121)
    title = "TRAIN-DATA\nLocal Weighted Lin Regressor Fn"
    plt.title(title)
    plt.scatter(train_data[train_sorted_ind], train_labels[train_sorted_ind])
    plt.plot(train_data[train_sorted_ind], np.array(y_pred_train)[train_sorted_ind])
    labels = ["data","local-wt lin-Reg"]
    plt.legend(labels)

    plt.subplot(122)
    title = "TEST-DATA\nLocal Weighted Lin Regressor Fn, and MSE = "+str(round(mse,4))
    plt.title(title)
    plt.scatter(test_data[test_sorted_ind], train_labels[test_sorted_ind])
    plt.plot(test_data[test_sorted_ind], np.array(y_pred_test)[test_sorted_ind])
    labels = ["data","local-wt lin-Reg"]
    plt.legend(labels)

    plt.show()
    
    """
    Part D explanation
    ------------------
    When we use only 20 data points for finding local weights, as mentioned in the problem,
    the test MSE increases significantly and becomes 12.8479, this can be because -
    - The data is too small
    - The weights of data points are not significant for getting general trend
    - The weights can be skewed in a direction to misalign the linear regressor
    - Hence more data points are needed
    """
    
    """
    Part E explanation
    ------------------
    After analysing the results from locally weighted linear regression model,
    I can say that it is likely that the data is generated from a function
    consistent with format of question 1, because -
    - The function from question 1 contains periodic functions of sine and cosine types
    - Hence the base function is likely to be periodic in nature
    - But the data definitely contains some noise other than periodicity
    - However, it is unclear if the noise is consistent with the base function or not
    - Because of high scattering, the chances are that the data comes from a complex function
    """
    
problem2(train_data, train_labels, test_data, test_labels)


# # ----------------------------------------------------------------------------------

# # Softmax Regression
# ---

# In[16]:


def format_train_test_data_so_re(dataset):
    """
    A function to properly structure the dataset given for 
    machine learning and split into features and labels.
    
    Parameters
    ----------
    dataset - list of str
    txt file loaded as working data
    
    Returns
    -------
    data - numpy array of float
    features to use for machine learning
    
    labels - numpy array of float
    target label to train and predict
    """
    
    f1 = []
    f2 = []
    f3 = []
    f4 = []
    labels = []

    for i in [i[:-2].split(',') for i in dataset]:
        f1.append(float(i[0]))
        f2.append(float(i[1]))
        f3.append(float(i[2]))
        f4.append(float(i[3]))
        if i[4][1:] == 'Plasti':
            labels.append('Plastic')
        else:
            labels.append(i[4][1:])
        
            
    return np.array([f1,f2,f3,f4]).T, np.array(labels)


# In[17]:


## load dataset
with open('train_softmax.txt') as f:
    train_data_so_re = f.readlines()
    f.close()
    
with open('test_softmax.txt') as f:
    test_data_so_re = f.readlines()
    f.close()

train_data_so_re, train_labels_so_re = format_train_test_data_so_re(train_data_so_re)
test_data_so_re, test_labels_so_re = format_train_test_data_so_re(test_data_so_re)


# In[18]:


## one hot encoding
labels = np.unique(train_labels_so_re)

# Create a dictionary to map the labels to integers
label_to_int = {label: i for i, label in enumerate(labels)}

# Use the dictionary to encode the data
train_labels_so_re_ohe = np.array([label_to_int[label] for label in train_labels_so_re])
test_labels_so_re_ohe = np.array([label_to_int[label] for label in test_labels_so_re])


# In[19]:


def soft_max_impl(row):
    """
    calculate softmax values for each row of input array
    """
    e_z = np.exp(row - np.max(row, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def model_fit(train_data, labels, learning_rate=0.01, iterations=10000):
    """
    Returns
    -------
    weight - numpy array of float
    model parameters calculated 
    """
    # one-hot encoding for the target label
    num_classes = len(set(labels))
    labels_onehot = np.eye(num_classes)[labels]

    # Add bias to the data
    train_data = np.concatenate((np.ones((train_data.shape[0], 1)), train_data), axis=1)

    # Initializing the weights
    Weight = np.zeros((train_data.shape[1], num_classes))

    # Gradient ascent
    for i in range(iterations):
        z = np.dot(train_data, Weight)
        y_pred = soft_max_impl(z)
        error = labels_onehot - y_pred
        delta_W = learning_rate * np.dot(train_data.T, error)
        Weight += delta_W

    return Weight

def model_predict(train_data, Weight):
    # including bias to the data
    train_data = np.concatenate((np.ones((train_data.shape[0], 1)), train_data), axis=1)

    # predicting probability for class
    z = np.dot(train_data, Weight)
    y_pred = np.argmax(soft_max_impl(z), axis=1)

    # return prediction
    return y_pred


# In[20]:


## execute model
weight = model_fit(train_data_so_re, train_labels_so_re_ohe)


# In[21]:


test_labels_so_re_ohe


# In[24]:


pred = model_predict(test_data_so_re, weight)


# In[26]:


accuracy = sum(pred == test_labels_so_re_ohe)/len(test_labels_so_re_ohe)

print('Accuracy is ',accuracy*100, "%")


# ## Part B
# ----
# Leave one out method for training

# In[28]:


## for calculating accuracy
total_correct = 0

for ind, row in enumerate(train_data_so_re[:-1]):
    ## remove elements one by one
    data_temp = train_data_so_re[ind]
    label_temp = train_labels_so_re_ohe[ind]
    
    train_data_temp = np.delete(train_data_so_re, ind, 0)
    train_label_temp = np.delete(train_labels_so_re_ohe, ind, 0)
    
    W = model_fit(train_data_temp, train_label_temp)
    
    pred_temp = model_predict(np.array([data_temp]), W)
    
    if pred_temp[0] == label_temp:
        total_correct += 1
        
print('Accuracy is ',100*(total_correct/len(train_data_so_re)))    


# As we can see, the accuracy in this case is greater than KNN method implemented in Homework 1.
# This is because the Softmax Regression is much better algorithm for this case of data.
# 
# Also, the leave one out method provides more data for training and testing can be done on each instance.
# Hence all the data points are covered to capture the variation in dataset.

# ## Part C
# ---
# Performance Without 4th feature 

# In[29]:


weight = model_fit(train_data_so_re[:,:-1], train_labels_so_re_ohe)


# In[31]:


y_pred = model_predict(test_data_so_re[:, :-1], weight)


# In[32]:


accuracy = sum(y_pred == test_labels_so_re_ohe)/len(test_labels_so_re_ohe)

print("accuracy is ", accuracy*100)


# """
# While analysing the accuracy, I found that 
# - this model is much better than KNN
# - Only the first 3 features capture the vital information for the prediction
# - I can say that the fourth feature is not contributing to the model's performance.
# - In fact, it seems that the 4th feature is deteriorating the performance
# - So, it is for good to remove it
# """
