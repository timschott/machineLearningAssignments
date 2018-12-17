# UVA CS 4501 Machine Learning- KNN assignment 3

__author__ = 'tcs9pk'

import numpy as np
import scipy.spatial as sp
import csv
import matplotlib.pyplot as plt

def read_csv(file):

    container = []
    # read in file
    with open(filename, 'r') as data:
        lines = csv.reader(data)
        next(lines, None)  # skip the headers
        for row in lines:
            container.append(row)

    # make floats not string
    container = np.array(container, float)

    np.random.seed(37)

    np.random.shuffle(container)

    return container


def fold(data, currenti):
    # split into training and testing based on what the current fold is.

    dup = np.concatenate((data, data), axis=0)
    i = currenti
    training = dup[(i - 1) * 500: (i - 1) * 500 + 1500, ]
    testing = dup[(i - 1) * 500 + 1500: (i - 1) * 500 + 2000, ]

    return training, testing

def classify(training, testing, k):
    # pull out the last column of training and testing
    sub_training = training[:, [0, 1]]
    sub_testing = testing[:, [0, 1]]
    training_last_column = training[:, -1]
    # calculate distance
    result = sp.distance.cdist(sub_testing, sub_training)
    labels = testing[:,-1]
    #each row in result is the distance from a testing point Xt to each training point X1, X2.. Xn.
    predictions = []
    for row in result:
        keys = row.tolist()
        vals = training_last_column.tolist()
        sorted_distances, sorted_labels = zip(*sorted(zip(keys, vals)))
        relevant = sorted_labels[0:k]
        composite = sum(relevant) / k
        if composite >= .5:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions, labels


def calc_accuracy(predictions, labs):

    residuals = np.subtract(predictions, labs)
    ssq = np.sum(residuals ** 2)
    ssq = ssq / 500

    return ssq

def findBestK(data, test_k, kfold):

    sum = 0
    error_list = []
    for k in test_k:
        sum = 0
        for i in range(1, 5):
            training, testing = fold(data, i)
            predictions, labels = classify(training, testing, k)
            sum += calc_accuracy(predictions, labels)
        accuracy = sum/kfold
        error_list.append(accuracy)

    # Graph the Results

    k = np.arange(1, 13, 2)
    error_list = np.array(error_list, 'float')
    plt.bar(k, error_list, align='center')
    plt.xlabel("K")
    plt.ylabel("MSE")
    plt.ylim(.45, .55)
    plt.xticks(k, ([str(thing) for thing in k]))
    plt.title('Number of Neighbors vs Error')
    plt.savefig('KNNError.png')
    plt.show()

    sorted_errors, sorted_k = zip(*sorted(zip(error_list, test_k)))

    return sorted_errors[0], sorted_k[0]


if __name__ == "__main__":
    filename = "Movie_Review_Data.csv"
    data = read_csv(filename)
    training, testing = fold(data, 4)
    pred, labels = classify(training,testing, 9)

    calc_accuracy(pred, labels)
    test_k = [1,3,5,7,9,11]
    best_error, best_k = findBestK(data, test_k, 4)

    print(best_error)
    print(best_k)