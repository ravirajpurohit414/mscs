import pandas as pd
import numpy as np
from data import training_data, testing_data, dataset_all, k_vals


## creating a function to generate dataset in a format
def create_dataset(data):
    """
    Parameters
    ----------
    data - generated data to be formatted

    Returns
    -------
    This function returns the formatted dataset
    """
    feature1 = []
    feature2 = []
    feature3 = []
    feature4 = []
    labels = []
    for i in data:
        feature1.append(i[0])
        feature2.append(i[1])
        feature3.append(i[2])
        feature4.append(i[3])
        labels.append(i[4])
    dataset = pd.DataFrame({'feature1': feature1, 'feature2': feature2, 'feature3': feature3, 'feature4': feature4, 'labels': labels})
    
    return dataset


def func_q2a():
    dataset = create_dataset(training_data)
    for item in testing_data:
        print("Testing data is - {}".format(item))
        for k in k_vals:
            print("K - {}".format(k))
            pred = knn_class(item, k, dataset, drop_feat_4=False, manhattan_bool=False)
            print("{} is the prediction for {} neighbors".format(pred, k))


def func_q2b():
    print("According to the announcement, execute part a for this")
    pass


def func_q2c():
    dataset_new = create_dataset(dataset_all)
    result = {}
    print("-----------------Output for Question 2c -----------------")
    for val in k_vals:
        val_preds_after_leave_one_out = 0
        for index, test_sample in dataset_new.iterrows():
            sample = test_sample.values[:4]
            target = test_sample.values[4]
            pred = knn_class(sample, val, dataset_new.drop(index), drop_feat_4=False, manhattan_bool=False)
            if target == pred:
                val_preds_after_leave_one_out += 1
            else:
                val_preds_after_leave_one_out += 0
        print("{}/{} true predictions with all features".format(val_preds_after_leave_one_out, dataset_new.shape[0]))
        result["When K = {}".format(val)] = val_preds_after_leave_one_out/len(dataset_new)*100
    print(result)


def func_q2d():
    print("-----------------Output for Question 2d -----------------")
    dataset_new = create_dataset(dataset_all)
    result = {}
    for val in k_vals:
        val_preds_after_leave_one_out = 0
        for index, test_sample in dataset_new.iterrows():
            sample = test_sample.values[:4]
            target = test_sample.values[4]
            pred = knn_class(sample, val, dataset_new.drop(index), drop_feat_4=False, manhattan_bool=True)
            if target == pred:
                val_preds_after_leave_one_out += 1
            else:
                val_preds_after_leave_one_out += 0
        print("{}/{} true predictions with all features".format(val_preds_after_leave_one_out, dataset_new.shape[0]))
        result["When K = {}".format(val)] = val_preds_after_leave_one_out/len(dataset_new)*100
    print(result)


def func_q2e():
    print("----------------- Output for Question 2e -----------------")
    dataset_new = create_dataset(dataset_all)
    result = {}
    for val in k_vals:
        valid_pred_exc_feat_4 = 0
        for _, test_item in dataset_new.iterrows():
            item = test_item.values[:4]
            target = test_item.values[4]
            pred = knn_class(item[:3], val, dataset_new, drop_feat_4=True, manhattan_bool=False)
            valid_pred_exc_feat_4 += 1 if target == pred else 0
        print("{}/{} true predictions excluding feature 4".format(valid_pred_exc_feat_4, dataset_new.shape[0]))
        result["When K = {}".format(val)] = valid_pred_exc_feat_4/len(dataset_new)*100
    print(result)


def cart_dist(sample, inps):
    """
    Parameters
    ----------
    sample - sample to process
    inps - dataset
    
    Returns
    -------
    This function returns the cartesian distances between the given dataset and sample
    """
    differences = sample - inps
    sum_of_pows = np.sum(np.power(differences, 2), axis=1)
    result = np.power(sum_of_pows, 0.5)
    
    return result


def calc_manhattan_dist(sample, inps):
    """
    Parameters
    ----------
    sample - a data item to work with
    inps - data inputs to calculate distance

    Returns
    -------
    This function returns the manhatten distance between the two inputs
    """
    differences = abs(sample - inps)
    result = np.sum(differences, axis=1)
    
    return result


def classifier(k, labels):
    """
    Parameters
    ----------
    k - number of neighbors
    labels - list of labels ordered by descending cartesian distance
    
    Returns
    -------
    This function performs the classification using KNN and returns the predicted class
    """
    k_n = labels[:k]
    metal_occs = np.count_nonzero(k_n == 'Metal')
    ceramic_occs = np.count_nonzero(k_n == 'Ceramic')
    plastic_occs = np.count_nonzero(k_n == 'Plastic')

    result = max_occ(metal_occs, ceramic_occs, plastic_occs)
    
    return result


def max_occ(metal, ceramic, plastic):
    """
    Parameters
    ----------
    metal - non zero counts of this type
    ceramic - non zero counts of this type
    plastic - non zero count of this type
    
    Returns
    -------
    This function returns the type of material which is mostly occured
    """
    if metal >= ceramic and metal >= plastic:
        return "Metal"
    elif ceramic >= metal and ceramic >= plastic:
        return "Ceramic"
    elif plastic >= metal and plastic >= ceramic:
        return "Plastic"


def knn_class(sample, k, dataset, drop_feat_4, manhattan_bool):
    
    ## drop feature 4 for one question
    if drop_feat_4:
        inps = dataset.drop(['feature4', 'labels'], axis=1).values
    else:
        inps = dataset.drop(['labels'], axis=1).values
    labels = dataset["labels"].values
    if(not manhattan_bool):
        cart_distance = cart_dist(sample, inps)
    else:
        cart_distance = calc_manhattan_dist(sample, inps)
    lab_cart = np.vstack((cart_distance, labels))
    ord_cart = lab_cart.T[lab_cart.T[:, 0].argsort()]
    labels = ord_cart.T[1]

    result = classifier(k, labels)
    
    return result


if __name__ == '__main__':
    
    print("executing ques 2a --------------")
    func_q2a()
    print("executing ques 2b --------------")
    func_q2b()
    print("executing ques 2c --------------")
    func_q2c()
    print("executing ques 2d --------------")
    func_q2d()
    print("executing ques 2e --------------")
    func_q2e()