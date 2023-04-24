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
    split the dataset into train-test randomly in 75:25 for training 
    and testing respectively
    
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
    np.random.seed(21)
    shuffled_indices = np.random.permutation(len(features))
    
    ## Split the shuffled indices into train and test sets
    train_indices = shuffled_indices[:int(len(features) * 0.75)]
    test_indices = shuffled_indices[int(len(features) * 0.75):]
    
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
    return np.argmax(A, axis=1)

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
# The results obtained show that the bagging algorithm improves the performance of a single classifier. As we can see from the results, the accuracy of a single classifier is 63.33%, while the accuracy of bagging with 10, 50, and 100 classifiers is 76.67%, 80.0%, and 80.0%, respectively. 
# 
# This increase in accuracy can be explained by the fact that bagging helps to reduce overfitting by training each classifier on a different subset of the data. By combining the predictions of multiple classifiers, we can reduce the variance of the model and improve its generalization performance on unseen data.
# 
# However, it's important to note that the results obtained may vary depending on the dataset and the choice of hyperparameters. In practice, it's recommended to perform a thorough hyperparameter tuning to achieve the best performance.

# In[ ]:





# ## ------------------------- Question 2 (A,B) -------------------------

# In[14]:


print('------------------------- Question 2 (A,B) -------------------------')


# In[15]:


## softmax function already defined above

def cross_entropy_loss(y_hat, y):
    """
    Computes the cross-entropy loss between the predicted probability 
    distribution and the true distribution of labels.
    
    Parameters
    ----------
    y_hat - numpy array of shape (m, k)
        Predicted probability distribution, where m is the number of samples 
        and k is the number of classes.
    y - numpy array of shape (m, k)
        True distribution of labels in one-hot encoding.
        
    Returns
    -------
    loss - float
        Cross-entropy loss value between y_hat and y.
    """
    m = y_hat.shape[0]
    loss = -np.sum(y * np.log(y_hat)) / m
    
    return loss

def one_hot_encode(y):
    """
    Converts the input array of labels to one-hot encoded matrix.

    Parameters
    ----------
    y - numpy array of shape (n_samples,)
        The input array of integer labels.

    Returns
    -------
    one_hot - numpy array of shape (n_samples, n_classes)
        The one-hot encoded matrix, where each row represents a sample and
        each column represents a class. The column corresponding to the 
        class of the sample contains 1, and all other columns contain 0.
    """
    n_values = np.max(y) + 1
    
    return np.eye(n_values)[y]

def softmax_regression(X, y, num_classes, num_iterations, learning_rate):
    """
    Trains a softmax regression model on the given input data X and labels y 
    to classify the samples into the specified number of classes using 
    stochastic gradient descent.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input features to be classified.
    y : numpy array of shape (n_samples,)
        The class labels corresponding to each input sample.
    num_classes : int
        The number of classes to classify the input samples into.
    num_iterations : int
        The number of iterations to train the model for.
    learning_rate : float
        The learning rate for the gradient descent optimization.

    Returns
    -------
    W : numpy array of shape (n_features, num_classes)
        The learned weight matrix for the softmax regression model.
    b : numpy array of shape (1, num_classes)
        The learned bias vector for the softmax regression model.
    """
    m, n = X.shape
    W = np.zeros((n, num_classes))
    b = np.zeros((1, num_classes))

    for i in range(num_iterations):
        Z = np.dot(X, W) + b
        A = softmax(Z)
        dZ = A - one_hot_encode(y)
        dW = np.dot(X.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)
        W -= learning_rate * dW
        b -= learning_rate * db

        if (i+1) % 1000 == 0:
            loss = cross_entropy_loss(A, one_hot_encode(y))
            print("Iteration %d, loss: %f" % (i+1, loss))

    return W, b

def adaboost(X, y, num_classes, num_iterations, learning_rate, num_boosts):
    """
    Trains an AdaBoost classifier by iteratively boosting multiple instances
    of a softmax regression classifier.

    Parameters
    ----------
    X : numpy array of shape (n_samples, n_features)
        The input feature matrix.

    y : numpy array of shape (n_samples,)
        The input array of integer labels.

    num_classes : int
        The number of classes in the classification problem.

    num_iterations : int
        The number of iterations to train the softmax regression classifier.

    learning_rate : float
        The learning rate used in the softmax regression classifier.

    num_boosts : int
        The number of iterations to boost the softmax regression classifier.

    Returns
    -------
    classifiers : list of tuples
        A list of tuples, where each tuple contains the weights and bias term
        of a trained softmax regression classifier.
    """
    m, n = X.shape
    weights = np.ones((m, 1)) / m
    classifiers = []

    for t in range(num_boosts):
        print("Boosting round %d" % (t+1))
        W, b = softmax_regression(X, y, num_classes, num_iterations, learning_rate)
        classifiers.append((W, b))

        y_hat = softmax(np.dot(X, W) + b)
        error = np.sum((y_hat.argmax(axis=1) != y)) / m
        alpha = 0.5 * np.log((1 - error) / error)

        y_pred = y_hat.argmax(axis=1)
        y_pred[y_pred != y] = -1
        y_pred[y_pred == y] = 1

        weights *= np.exp(-alpha * y_pred.reshape(m, 1))
        weights /= np.sum(weights)

    return classifiers

def predict(X, classifiers):
    """
    Takes in a set of input data samples and a set of trained classifiers 
    as input and returns the predicted class labels for each sample.

    Parameters
    ----------
    X: numpy array of shape (n_samples, n_features)
        The input data samples to be classified.
    
    classifiers: list of tuples, each tuple containing:
    
    W: numpy array of shape (n_features, n_classes)
        The weights learned by the softmax regression classifier.
    
    b: numpy array of shape (1, n_classes)
        The biases learned by the softmax regression classifier.
    
    Returns
    -------
    y_hat: numpy array of shape (n_samples,)
        The predicted class labels for each input data sample in X.
    """
    y_hat = np.zeros((X.shape[0], classifiers[0][0].shape[1]))

    for W, b in classifiers:
        y_hat += softmax(np.dot(X, W) + b)

    return y_hat.argmax(axis=1)


# In[16]:


y_train = np.array([np.argmax(i) for i in train_labels])
y_test = np.array([np.argmax(i) for i in test_labels])

## Train softmax regression classifier
W, b = softmax_regression(train_features, 
                          y_train, 
                          num_classes=3, 
                          num_iterations=1000, 
                          learning_rate=0.01)

## Evaluate single classifier
y_pred = predict(test_features, [(W, b)])
error_rate_single = np.sum((y_pred != y_test)) / y_test.shape[0]


# In[17]:


## Train AdaBoost ensemble
classifiers_10 = adaboost(train_features, 
                          y_train, 
                          num_classes=3, 
                          num_iterations=10000, 
                          learning_rate=0.01, 
                          num_boosts=10)

classifiers_25 = adaboost(train_features, 
                          y_train, 
                          num_classes=3, 
                          num_iterations=10000, 
                          learning_rate=0.01, 
                          num_boosts=25)

classifiers_50 = adaboost(train_features, 
                          y_train, 
                          num_classes=3, 
                          num_iterations=10000, 
                          learning_rate=0.01, 
                          num_boosts=50)

## Evaluate AdaBoost ensembles
y_pred_10 = predict(test_features, classifiers_10)
error_rate_10 = np.sum((y_pred_10 != y_test)) / y_test.shape[0]

y_pred_25 = predict(test_features, classifiers_25)
error_rate_25 = np.sum((y_pred_25 != y_test)) / y_test.shape[0]

y_pred_50 = predict(test_features, classifiers_50)
error_rate_50 = np.sum((y_pred_50 != y_test)) / y_test.shape[0]


# In[18]:


print(f"Single Classifier Error Rate: {round(100*error_rate_single,2)}%")
print(f"Boosting 10 Error Rate: {round(100*error_rate_10,2)}%")
print(f"Boosting 50 Error Rate: {round(100*error_rate_25,2)}%")
print(f"Boosting 100 Error Rate: {round(100*error_rate_50,2)}%")


# ## Observations -
# The results of the evaluation show that the AdaBoost ensembles significantly outperformed the single classifier. The single classifier had an error rate of 23.3%, while the AdaBoost ensembles with 10, 25, and 50 boosting rounds all had an error rate of only 13.3%. This is a substantial improvement in accuracy, indicating that AdaBoost is a powerful technique for improving the performance of machine learning models.
# 
# Additionally, we can see that the performance of the AdaBoost ensembles does not seem to improve beyond 25 boosting rounds, as the error rate remains constant at 13.3% for 25 and 50 boosting rounds. This suggests that further boosting may not be necessary and could potentially lead to overfitting.

# ## ------------------------- Question 3 (A,B) -------------------------

# In[19]:


print('------------------------- Question 3 (A,B) -------------------------')


# In[20]:


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


# In[21]:


## Scale the data
features_scaled = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

## Apply K-Means Clustering with K=3, 6, and 9
print('Before scaling the features -----')
for k in [3, 6, 9]:
    centroids, cluster_assignments = k_means_clustering(features, k)
    accuracy = compute_cluster_accuracy(cluster_assignments, 
                                        np.array([0 if i=='Plastic' else 1 if i=='Ceramic' else 2 for i in labels]), 
                                        k)
    print(f"K-Means Clustering, k = {k}, Overall Accuracy = {round(100*accuracy,2)}%")
    
print('\nAfter scaling the features -----')
## use scaled data to compare performance
## Apply K-Means Clustering with K=3, 6, and 9
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
# With K=6, the overall accuracy increases to 47.5%. This suggests that some of the overlap between the different material types is being captured by the algorithm, but there is still some confusion between the different clusters.
# 
# Finally, with K=9, the overall accuracy increases further to 55.8%. This suggests that the additional clusters are helping to better capture the different material types and reduce the confusion between them.
# 
# Overall, the results suggest that K-Means clustering can be effective at identifying the different material types in the dataset, but that a larger number of clusters may be needed to achieve high accuracy. Additionally, it's worth noting that the accuracy of K-Means clustering is limited by the intrinsic separability of the data, which may not be perfect in all cases.
# 
# ### After scaling the features -  
# Scaling the data had a significant impact on the performance of K-Means clustering. The overall accuracy increased for all values of K, which suggests that scaling improved the clustering results.
# 
# In particular, the accuracy of the K-Means clustering (K=9) increased from 55.8% to 75.8% after scaling. This is a substantial improvement and indicates that the clusters are better aligned with the true labels.
# 
# Scaling is an important preprocessing step for many machine learning algorithms, as it can help improve the performance and stability of the models. In this case, scaling helped K-Means clustering to better capture the structure of the data and produce more accurate clusters.

# In[ ]:





# ## References -
# https://towardsdatascience.com/ml-from-scratch-logistic-and-softmax-regression-9f09f49a852c
# 
# https://www.geeksforgeeks.org/bagging-vs-boosting-in-machine-learning/#
# 
# https://machinelearningmastery.com/implement-bagging-scratch-python/

# In[ ]:





# In[ ]:




