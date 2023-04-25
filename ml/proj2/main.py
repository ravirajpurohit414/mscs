#!/usr/bin/env python
# coding: utf-8

# # ------------------------- CSE-6363-001 ML Project 2 -------------------------

# In[1]:


## import libraries
import numpy as np


# ## ------------------------- Loading Data and Preprocessing -------------------------

# In[2]:


def read_data(filename='./data/data.txt'):
    """
    Load the dataset provided with the homework.
    
    Parameters
    ----------
    filename - string
    
    Returns
    -------
    data - numpy array of floats
    labels - numpy array of integers
    """
    features = []
    labels = []
    
    with open(filename, 'r') as f:
        for line in f:
            ## remove noise from the row of data and separate features and labels
            line = line.strip().replace('\n','').split(',')
            features.append([float(i) for i in line[:4]])
            labels.append(line[-1])
            
    return np.array(features), np.array(labels)


# In[3]:


features, labels = read_data()
print(f"shape of features - {features.shape}, \nshape of labels - {labels.shape}")


# In[4]:


print(f'\nlabel categories are - {[i for i in np.unique(labels)]}\n')


# In[5]:


def one_hot_encoding(labels):
    """
    perform one hot encoding to the labels
    
    Parameters
    ----------
    labels - numpy array of strings
    
    Returns
    -------
    labels_encoded - numpy array of one-hot-encoded labels
    
    """
    num_samples = len(labels)
    labels_encoded = np.zeros((num_samples, len(np.unique(labels))))
    for i in range(num_samples):
        if labels[i] == 'Plastic':
            labels_encoded[i, 0] = 1
        elif labels[i] == 'Metal':
            labels_encoded[i, 1] = 1
        elif labels[i] == 'Ceramic':
            labels_encoded[i, 2] = 1
            
    return labels_encoded


# In[6]:


labels_enc = one_hot_encoding(labels)
print(labels_enc[:5])


# In[7]:


def train_test_split(features, labels):
    """
    split the dataset into train-test randomly in 70:30 for 
    training and testing respectively
    
    Parameters
    ----------
    features - numpy array of floats
    labels - numpy array of strings
    
    Returns
    -------
    train_features - numpy array of floats
    test_features - numpy array of floats
    train_labels - numpy array of strings
    test_labels - numpy array of strings
    
    """
    ## Shuffle the indices
    np.random.seed(24)
    shuffled_indices = np.random.permutation(len(features))
    
    ## Split the shuffled indices into train and test sets
    train_indices = shuffled_indices[:int(len(features) * 0.70)]
    test_indices = shuffled_indices[int(len(features) * 0.70):]
    
    ## Use the train and test indices to split the features and labels
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    
    return train_features, train_labels, test_features, test_labels


# In[8]:


train_features, train_labels, test_features, test_labels = train_test_split(features, labels_enc)


# In[9]:


print(f'shape of training dataset: features - {train_features.shape}, labels - {train_labels.shape}')
print(f'shape of testing dataset: features - {test_features.shape}, labels - {test_labels.shape}')


# In[ ]:





# ## ------------------------- Question 1 (A,B) -------------------------

# In[10]:


print('------------------------- Question 1 (A,B) -------------------------')


# In[11]:


def softmax(i):
    """
    Applies the softmax function element-wise to each row of the input array.
    
    Parameters
    ----------
    i - a numpy array of shape (n_samples, n_classes) containing the output 
    of a linear transformation of the input data.

    Returns
    -------
    res - a numpy array of shape (n_samples, n_classes) containing the 
    probabilities of each sample belonging to each class, 
    computed using the softmax function.
    """
    e_i = np.exp(i - np.max(i, axis=1, keepdims=True))
    res = e_i / e_i.sum(axis=1, keepdims=True)

    return res

def propagate(features, labels, W, b):
    """
    Computes forward propagation and backward propagation for 
    a softmax regression model.

    Parameters
    ----------
    features - numpy array of shape (m, n)
        The input data, where m is the number of samples and 
        n is the number of features.
    labels - numpy array of shape (m, c)
        The one-hot encoded target labels, where c is the number of classes.
    W - numpy array of shape (n, c)
        The weight matrix for the linear transformation.
    b - numpy array of shape (1, c)
        The bias vector.

    Returns
    -------
    dW - numpy array of shape (n, c)
        The gradient of the cost function with respect to W.
    db - numpy array of shape (1, c)
        The gradient of the cost function with respect to b.
    cost - float
        The value of the cost function.
    """
    m = features.shape[0]
    A = softmax(np.dot(features, W) + b)
    cost = -np.mean(np.sum(labels * np.log(A), axis=1))
    dZ = A - labels
    dW = (1 / m) * np.dot(features.T, dZ)
    db = (1 / m) * np.sum(dZ, axis=0)

    return dW, db, cost

def optimize(features, labels, W, b, num_iter, lr):
    """
    Performs gradient descent optimization to minimize the cross-entropy loss
    between the predicted and actual class probabilities.

    Parameters
    ----------
    features - numpy array of shape (n_samples, n_features)
        The input data.
    labels - numpy array of shape (n_samples, n_classes)
        The one-hot encoded target labels.
    W - numpy array of shape (n_features, n_classes)
        The weight matrix of the softmax regression classifier.
    b - numpy array of shape (1, n_classes)
        The bias vector of the softmax regression classifier.
    num_iter - int
        The number of iterations to run the optimization algorithm.
    lr - float
        The learning rate, which controls the step size of the parameter updates.

    Returns
    -------
    W - numpy array of shape (n_features, n_classes)
        The optimized weight matrix.
    b - numpy array of shape (1, n_classes)
        The optimized bias vector.
    costs - list
        A list of the cross-entropy losses at every 100 iterations of the
        optimization algorithm.
    """
    costs = []
    for i in range(num_iter):
        dW, db, cost = propagate(features, labels, W, b)
        W -= lr * dW
        b -= lr * db
        if i % 100 == 0:
            costs.append(cost)

    return W, b, costs

def predict(features, W, b):
    """
    Predicts the class label for each sample in X, based on the 
    learned parameters W and b.

    Parameters
    ----------
    features - numpy array of shape (n_samples, n_features)
        The input data.

    W - numpy array of shape (n_features, n_classes)
        The learned weights for the linear transformation.

    b - numpy array of shape (1, n_classes)
        The learned bias terms for the linear transformation.

    Returns
    -------
    predictions - numpy array of shape (n_samples,)
        The predicted class label for each sample in X, as an 
        integer between 0 and (n_classes - 1).
    """
    A = softmax(np.dot(features, W) + b)
    predictions = np.argmax(A, axis=1)

    return predictions

def bagging(train_features, train_labels, test_features, test_labels, num_bagging):
    """
    Applies bagging to train a model on different subsets of 
    the training data and then aggregates their predictions 
    to make a final prediction.

    Parameters
    ----------
    train_features - numpy array of shape (n_train_samples, n_features)
        The features of the training data.
    train_labels - numpy array of shape (n_train_samples, n_classes)
        The one-hot encoded labels of the training data.
    test_features - numpy array of shape (n_test_samples, n_features)
        The features of the test data.
    test_labels - numpy array of shape (n_test_samples, n_classes)
        The one-hot encoded labels of the test data.
    num_bagging - int
        The number of subsets to create and train models on.

    Returns
    -------
    pred - numpy array of shape (n_test_samples,)
        The predicted labels for the test data.
    """
    bagging_pred = np.zeros((test_labels.shape[0], num_bagging))
    for i in range(num_bagging):
        idx = np.random.choice(train_features.shape[0], train_features.shape[0])
        X_bag, y_bag = train_features[idx], train_labels[idx]
        ## initialize parameters
        W = np.zeros((train_features.shape[1], 3))
        b = np.zeros((1, 3))
        W, b, _ = optimize(X_bag, y_bag, W, b, num_iter=10000, lr=0.1)
        bagging_pred[:, i] = predict(test_features, W, b)
    
    pred = np.argmax(np.apply_along_axis(lambda x: np.bincount(x.astype('int64'), 
                                                               minlength=3), 
                                         axis=1, arr=bagging_pred), axis=1)

    return pred


# In[12]:


## run the bagging algorithm for single, 10, 50 and 100 cases
pred_1 = bagging(train_features, 
                 train_labels, 
                 test_features, 
                 test_features, 
                 num_bagging=1)

pred_10 = bagging(train_features, 
                  train_labels, 
                  test_features, 
                  test_features, 
                  num_bagging=10)

pred_50 = bagging(train_features, 
                  train_labels, 
                  test_features, 
                  test_features, 
                  num_bagging=50)

pred_100 = bagging(train_features, 
                   train_labels, 
                   test_features, 
                   test_features, 
                   num_bagging=100)

## convert actual label to normal encoded labels for performance comparison
pred_actual = np.array([np.argmax(i) for i in test_labels])


# In[13]:


print(f"Single Classifier Accuracy: {round(100*np.mean(pred_1==pred_actual),2)}%")
print(f"Bagging 10 Accuracy: {round(100*np.mean(pred_10==pred_actual),2)}%")
print(f"Bagging 50 Accuracy: {round(100*np.mean(pred_50==pred_actual),2)}%")
print(f"Bagging 100 Accuracy: {round(100*np.mean(pred_100==pred_actual),2)}%")

print('\n')

print(f"Single Classifier Error Rate: {100-round(100*np.mean(pred_1==pred_actual),2)}%")
print(f"Bagging 10 Error Rate: {100-round(100*np.mean(pred_10==pred_actual),2)}%")
print(f"Bagging 50 Error Rate: {100-round(100*np.mean(pred_50==pred_actual),2)}%")
print(f"Bagging 100 Error Rate: {100-round(100*np.mean(pred_100==pred_actual),2)}%")


# ## Observations -
# 
# The results indicate that the single classifier has an accuracy of 83.33% and an error rate of 16.67%. When using bagging with 10 iterations, the accuracy decreases slightly to 80.56% and the error rate increases to 19.44%. However, when using bagging with 50 or 100 iterations, the accuracy remains the same as the single classifier at 83.33%, and the error rate also remains the same at 16.67%.
# 
# This suggests that bagging with a larger number of iterations can help improve the accuracy of the model while reducing the variance. However, if the number of iterations is too small, it may lead to overfitting and decrease the model's accuracy. Overall, the results indicate that bagging can be a useful technique to improve the accuracy and robustness of machine learning models.
# 
# But for a different random state, I got the results as - the accuracy of a single classifier is 63.33%, while the accuracy of bagging with 10, 50, and 100 classifiers is 76.67%, 80.0%, and 80.0%, respectively. This increase in accuracy can be explained by the fact that bagging helps to reduce overfitting by training each classifier on a different subset of the data. By combining the predictions of multiple classifiers, we can reduce the variance of the model and improve its generalization performance on unseen data.
# 
# However, it's important to note that the results obtained may vary depending on the dataset and the choice of hyperparameters. In practice, it's recommended to perform a thorough hyperparameter tuning to achieve the best performance.

# In[ ]:





# ## ------------------------- Question 2 (A,B) -------------------------

# In[14]:


print('------------------------- Question 2 (A,B) -------------------------')


# In[15]:


## softmax function already defined above

def softmax_regression(features, labels, n_cats, lr=0.1, iterations=10000):
    """
    Trains a softmax regression model using gradient descent optimization. 
    It takes in the input features, one-hot encoded labels, 
    number of categories, learning rate, and number of iterations.

    Parameters
    ----------

    features - a numpy array of shape (n_samples, n_features) representing 
        the features or input data for training the softmax regression model.

    labels -  a numpy array of shape (n_samples, n_cats) representing the 
        one-hot encoded labels for the input data.

    n_cats - an integer representing the number of categories or classes in the dataset.
    
    lr - a float representing the learning rate used for gradient 
        descent optimization. Default value is 0.1
    
    iterations - an integer representing the number of iterations or iterations 
        for training the model. Default value is 10000.
    
    Returns
    -------
    weights - a numpy array of shape (n_features, n_cats) representing the 
        learned weights or coefficients of the softmax regression model.
    bias - a numpy array of shape (1, n_cats) representing the learned bias 
        or intercept term of the softmax regression model.
    """
    
    n_samples, n_features = features.shape
    weights = np.zeros((n_features, n_cats))
    bias = np.zeros((1, n_cats))

    for epoch in range(iterations):
        labels_pred = softmax(np.dot(features, weights) + bias)
        error = labels_pred - labels
        gradient = np.dot(features.T, error)
        weights -= lr * gradient
        bias -= lr * np.sum(error, axis=0, keepdims=True)

    return weights, bias

def predict(features, weights, bias):
    """
    Parameters
    ----------
    features - 2D numpy array containing the input features for which 
        the class labels need to be predicted
    
    weights - 2D numpy array containing the learned weights of the 
        softmax regression classifier
    
    bias - 2D numpy array containing the learned bias terms of the 
        softmax regression classifier

    Returns
    -------
    predictions - 1D numpy array containing the predicted class labels 
        for each input feature in features. The predicted class label is 
        determined by selecting the class with the highest predicted 
        probability as calculated by the softmax function.
    """
    dot_prod = np.dot(features, weights) + bias
    predictions = np.argmax(softmax(dot_prod), axis=1)

    return predictions

def update_prediction(test_features, n_cats, clfs):
    """
    This function updates the prediction for a given set of test features 
    by applying the ensemble of classifiers created through boosting.

    Parameters
    ----------
    test_features - a numpy array of shape (n_samples, n_features) 
        representing the test features.
    
    n_cats - an integer representing the number of classes in the dataset.
    
    clfs - a list of tuples, where each tuple contains the weights and bias 
        learned by the softmax regression classifier.

    Returns
    -------
    pred_updated - a numpy array of shape (n_samples,) containing the updated 
        predictions for the test features, where each element corresponds to 
        the predicted category for the corresponding test feature.
    """
    pred = np.zeros((test_features.shape[0], n_cats))
    for weights, bias in clfs:
        pred += softmax(np.dot(test_features, weights) + bias)
    pred_updated = np.argmax(pred, axis=1)
    
    return pred_updated

def execute_boost(train_features, train_labels, test_features, test_labels):
    """
    execute the boosting technique on the dataset to compare the performances
    
    Parameters
    ----------
    train_features - numpy array of training features
    
    train_labels - numpy array of one-hot encoded labels
    
    test_features - numpy array of testing features
    
    test_labels - numpy array of one-hot encoded labels
    """
    train_features, test_features = train_features, test_features
    train_labels, test_labels = train_labels, test_labels
    size_total_samples = train_features.shape[0]
    n_cats = 3

    ## Train a single softmax regression classifier
    weights, bias = softmax_regression(train_features, train_labels, n_cats=3)
    y_pred = predict(test_features, weights, bias)
    error_rate_single = np.mean(y_pred != np.argmax(test_labels, axis=1))

    ## Train an ensemble of softmax regression clfs using boosting
    boost_ops = [10, 25, 50]
    ## populate a list for collecting error rates
    error_rates = []
    for boost_op in boost_ops:
        clfs = []
        i = 0
        while i < boost_op:
            ## generate samples weights and indices
            s_wts = np.ones(size_total_samples) / size_total_samples
            s_inds = np.random.choice(range(size_total_samples), 
                                              size_total_samples, 
                                              replace=True, 
                                              p=s_wts)
            X_s = train_features[s_inds]
            y_s = train_labels[s_inds]
            weights, bias = softmax_regression(X_s, y_s, 3)
            clfs.append((weights, bias))

            y_pred = update_prediction(test_features, n_cats, clfs)
            
            accuracy = np.mean(y_pred == np.argmax(test_labels, axis=1))
            i += 1
        error_rates.append(1-accuracy)

    # Print and compare the results
    print(f"Single Classifier Error Rate: {round(100*error_rate_single,2)}%")
    for error, boost_op in zip(error_rates, boost_ops):
        print(f"Boosting {boost_op} Error Rate: {round(100*error,2)}%")        


# In[16]:


execute_boost(train_features, train_labels, test_features, test_labels)


# In[ ]:





# ## Observations -
# The results of the evaluation show that the AdaBoost ensembles significantly outperformed the single classifier. The single classifier had an error rate of 22.22%, while the AdaBoost ensembles with 10, 25, and 50 boosting rounds all had an error rate of ~15%. This is a substantial improvement in accuracy, indicating that AdaBoost is a powerful technique for improving the performance of machine learning models.
# 
# Additionally, we can see that the performance of the AdaBoost ensembles does not seem to significantly improve beyond 25 boosting rounds, as the error rate remains consistent for 25 and 50 boosting rounds. This suggests that further boosting may not be necessary and could potentially lead to overfitting.

# ## ------------------------- Question 3 (A,B) -------------------------

# In[17]:


print('------------------------- Question 3 (A,B) -------------------------')


# In[35]:


def k_means_clustering(X, k, num_iterations=100):
    """
    Implements the k-means clustering algorithm on a given 
    dataset to identify k clusters.

    Parameters
    ----------
    X - numpy array of shape (n_samples, n_features)
        The input data matrix.

    k - int
        The number of clusters to identify.

    num_iterations - int, optional (default=100)
        The maximum number of iterations to run the algorithm for.

    Returns
    -------
    centroids - numpy array of shape (k, n_features)
        The final centroids of the k clusters.

    cluster_assignments - numpy array of shape (n_samples,)
        An array containing the cluster assignments of each point in X.
    """
    ## Randomly initializing centroids to begin
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]

    # Iterate until max number of iterations or convergence is achieved
    for i in range(num_iterations):
        ## Assign each point to its nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)
        cluster_assignments = np.argmin(distances, axis=1)

        ## Update centroids
        for j in range(k):
            mask = (cluster_assignments == j)
            if np.any(mask):
                centroids[j] = np.mean(X[mask, :], axis=0)

    return centroids, cluster_assignments

def compute_cluster_accuracy(cluster_assignments, true_labels, k):
    """
    Computes the accuracy of each cluster assignment and returns 
    the average accuracy across all clusters.

    Parameters
    ----------
    cluster_assignments - numpy array of shape (n_samples,)
        An array containing the cluster assignments for each sample.

    true_labels - numpy array of shape (n_samples,)
        An array containing the true labels for each sample.

    k - int
        The number of clusters.

    Returns
    -------
    avg_accuracy - float
        The average accuracy of each cluster assignment across all clusters.
    """
    accuracies = []
    for j in range(k):
        mask = (cluster_assignments == j)
        if np.any(mask):
            counts = np.bincount(true_labels[mask])
            accuracy = np.max(counts) / np.sum(counts)
            accuracies.append(accuracy * np.sum(mask) / true_labels.shape[0])
    avg_accuracy = np.sum(accuracies)
    
    return avg_accuracy


# In[36]:


## Scale the data
features_scaled = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

## Applying K-Means Clustering with K=3, 6, and 9
print('Before scaling the features -----')
for k in [3, 6, 9]:
    centroids, cluster_assignments = k_means_clustering(features, k)
    accuracy = compute_cluster_accuracy(cluster_assignments, 
                                        np.array([0 if i=='Plastic' else 1 if i=='Ceramic' else 2 for i in labels]), 
                                        k)
    print(f"K-Means Clustering, k = {k}, Overall Accuracy = {round(100*accuracy,2)}%")
    
print('\nAfter scaling the features -----')
## use scaled data to compare performance
for k in [3, 6, 9]:
    centroids, cluster_assignments = k_means_clustering(features_scaled, k)
    accuracy = compute_cluster_accuracy(cluster_assignments, 
                                        np.array([0 if i=='Plastic' else 1 if i=='Ceramic' else 2 for i in labels]), 
                                        k)
    print(f"K-Means Clustering, k = {k}, Overall Accuracy = {round(100*accuracy,2)}%")


# ## Observations -
# ### Before scaling the features -
# The results of the K-Means clustering algorithm show that the accuracy increases with an increase in the number of clusters, as expected. With K=3, the overall accuracy of the algorithm is 43.33%, which is not very high. This is likely due to the fact that there are three distinct material types in the dataset, which may not be easily separable into just three clusters.
# 
# With K=6, the overall accuracy increases to ~46%. This suggests that some of the overlap between the different material types is being captured by the algorithm, but there is still some confusion between the different clusters.
# 
# Finally, with K=9, the overall accuracy increases further to ~55%. This suggests that the additional clusters are helping to better capture the different material types and reduce the confusion between them.
# 
# Overall, the results suggest that K-Means clustering can be effective at identifying the different material types in the dataset, but that a larger number of clusters may be needed to achieve high accuracy. Additionally, it's worth noting that the accuracy of K-Means clustering is limited by the intrinsic separability of the data, which may not be perfect in all cases.
# 
# ### After scaling the features -  
# Scaling the data had a significant impact on the performance of K-Means clustering. The overall accuracy increased for all values of K, which suggests that scaling improved the clustering results.
# 
# In particular, the accuracy of the K-Means clustering (K=9) increased from 55% to 74% after scaling. This is a substantial improvement and indicates that the clusters are better aligned with the true labels.
# 
# Scaling is an important preprocessing step for many machine learning algorithms, as it can help improve the performance and stability of the models. In this case, scaling helped K-Means clustering to better capture the structure of the data and produce more accurate clusters.

# In[ ]:





# ## References -
# I have referred to the following articles in order to understand the nuances of softmax regression, bagging and boosting techniques.
# 
# https://towardsdatascience.com/ml-from-scratch-logistic-and-softmax-regression-9f09f49a852c
# 
# https://www.geeksforgeeks.org/bagging-vs-boosting-in-machine-learning/#
# 
# https://machinelearningmastery.com/implement-bagging-scratch-python/

# In[ ]:




