#!/usr/bin/python
__author__ = 'tcs9pk'

import numpy as np
import csv
import sys
from copy import deepcopy
import matplotlib.pyplot as plt


def loadData(fileDj):
    dummy = []
    height = []
    weight = []

    # read in file
    with open(fileDj, 'r') as data:
        lines = csv.reader(data, delimiter=' ')
        for heights, weights, dummies in lines:
            height.append(heights)
            weight.append(weights)
            dummy.append(dummies)

    # make floats not strings

    height = [float(i) for i in height]
    weight = [float(i) for i in weight]
    labels = [float(i) for i in dummy]

    height = np.array(height, float)
    weight = np.array(weight, float)

    # normalize our data.
    # this greatly improved my performance.

    for i in range(len(height)):
        height[i] = (height[i] - np.amin(height)) / (np.amax(height) - np.amin(height))
        weight[i] = (weight[i] - np.amin(weight)) / (np.amax(weight) - np.amin(weight))

    labels = np.array(labels, float)

    # put into one, well-formed 3 column numpy array

    test = np.concatenate((height.reshape(-1,1), weight.reshape(-1,1)), axis=1)
    data = np.concatenate((test, labels.reshape(-1,1)), axis=1)

    return data


## K-means functions 

def getInitialCentroids(X, k):

    height= X[:, 0]
    weight = X[:, 1]

    height_list = []
    weight_list = []

    np.random.seed(32)

    for i in range(0, k):
        # randomly initalize height and weight centroids.
        # this loop gives us "k" random pairs of height and weight
        index = np.random.choice(height.shape[0], replace=False)
        # we pull from the same index so their values match up
        height_list.append(height[index])
        weight_list.append(weight[index])


    height_list = np.array(height_list, float)
    weight_list = np.array(weight_list, float)

    initialCentroids = np.concatenate((height_list.reshape(-1, 1), weight_list.reshape(-1, 1)), axis=1)

    return initialCentroids


def find_distance(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


def allocatePoints(X, centroids):

    # for each pair of points in X, assign it to the cluster that has the smallest distance

    data = X[:, [0,1]]
    # cluster container
    clusters = []

    for i in range(len(X)):
        # calculate distance
        distances = find_distance(data[i], centroids)
        # pick smallest
        cluster = np.argmin(distances)
        # add
        clusters.append(cluster)

    clusters = np.array(clusters, float)

    return clusters

def updateCentroids(X, clusters, k):

    # make containers for height and weight
    height_list = []
    weight_list = []


    for i in range(k):
        # if applicable, add average
        points = [X[n] for n in range(len(X)) if clusters[n] == i]
        height_list.append(np.mean(points, axis=0)[0])
        weight_list.append(np.mean(points, axis=0)[1])

    # turn into np array
    weight_list = np.array(weight_list, float)
    height_list = np.array(height_list, float)

    # new centroids
    centroids = np.concatenate((height_list.reshape(-1, 1), weight_list.reshape(-1, 1)), axis=1)

    return centroids

def visualizeClusters(X, clusters):
    # pull out heights
    heights = X[:, [0]]
    weights = X[:, [1]]

    heights = heights.tolist()
    weights = weights.tolist()

    # np array so it can be used in matplot lib
    heights_np = np.array(heights, float)
    weights_np = np.array(weights, float)

    # now we plot.
    # can leave as a single map, if K < 6 some colors just won't get used.
    colors = {0.0: 'red', 1.0: 'blue', 2.0: 'green', 3.0: 'm', 4.0:'cyan', 5.0: 'yellow'}

    label_color = [colors[l] for l in clusters]
    plt.scatter(weights_np, heights_np, c=label_color)
    plt.xlabel("weights")
    plt.ylabel("heights")
    plt.title('Clustering with K = 6')
    plt.savefig('cluster_56png')
    plt.show()

    #Your code here
    return 0

def kmeans(X, k, maxIter):

    # initial work - create containers for old centroids and grab the first ones.
    centroids = getInitialCentroids(X, k)
    centroids_old = np.zeros(centroids.shape)
    # error is an array because we can have multiply centroids
    error = find_distance(centroids, centroids_old)

    for i in range(0, maxIter):
        # get clusters
        clusters = allocatePoints(X, centroids)
        # copy over to compare later
        centroids_old = deepcopy(centroids)
        centroids = updateCentroids(X, clusters, k)
        error = find_distance(centroids, centroids_old)
        # if everything in the error array is zero, get out of there
        if(np.count_nonzero(error)):
            break

    # return all these for the auxiliary methods- viz, purity checking and kneefinding
    return centroids, clusters, error


# more helpful than one would think. cf purity tests
def dumbDivision(a, b):
    if b == 0:
        return 0
    else:
        return a/b

def kneeFinding(X,kList):

    # run k means for k = 0 - 5

    #Your code here
    mse_list = []
    for i in range(len(kList)):
        centroids, clusters, error = kmeans(X, kList[i], maxIter=1000)
        mse = np.mean(error)
        mse_list.append(mse)

    k = kList
    error_list = np.array(mse_list, 'float')
    plt.plot(k, error_list)
    plt.xlabel("K")
    plt.ylabel("MSE")
    plt.xticks(k, ([str(thing) for thing in k]))
    plt.title('Elbowfinding with KMeans')
    plt.savefig('ElbowError.png')
    plt.show()

    return None

def purity(X, clusters):

    labels = X[:, [2]]
    labels = np.array(labels, float)

    purities = []

    clusters_list = clusters.tolist()
    labels_list = labels.tolist()

    sorted_clusters, sorted_labels = zip(*sorted(zip(clusters_list, labels_list)))

    sorted_labels = [item for sublist in sorted_labels for item in sublist]

    zero_count = 0
    zero_total = 0

    one_count = 0
    one_total = 0

    two_count = 0
    two_total = 0

    three_count = 0
    three_total = 0

    four_count = 0
    four_total = 0

    five_count = 0
    five_total = 0

    # an extremely naive way of calculating purities..

    for i in range(len(sorted_labels)):
        if sorted_clusters[i] == 0.0:
            if sorted_labels[i] == 1.0:
                zero_count += 1
            zero_total += 1

        if sorted_clusters[i] == 1.0:
            if sorted_labels[i] == 1.0:
                one_count += 1
            one_total += 1

        if sorted_clusters[i] == 2.0:
            if sorted_labels[i] == 1.0:
                two_count += 1
            two_total += 1

        if sorted_clusters[i] == 3.0:
            if sorted_labels[i] == 1.0:
                three_count += 1
            three_total += 1

        if sorted_clusters[i] == 4.0:
            if sorted_labels[i] == 1.0:
                four_count += 1
            four_total += 1

        if sorted_clusters[i] == 5.0:
            if sorted_labels[i] == 1.0:
                five_count += 1
            five_total += 1


    # add in all purities, even if no error was encountered
    # this allows me to just run this method no matter what the k is
    purities.append(max(zero_count, zero_total-zero_count)/float(zero_total))
    purities.append(dumbDivision(max(one_count, (one_total - one_count)), float(one_total)))
    purities.append(dumbDivision(max(two_count, (two_total- two_count)), float(two_total)))
    purities.append(dumbDivision(max(three_count, (three_total - three_count)), float(three_total)))
    purities.append(dumbDivision(max(four_count, four_total - four_count), float(four_total)))
    purities.append(dumbDivision(max(five_count, five_total - five_count), float(five_total)))

    purities[:] = (value for value in purities if value != 0.0)

    return purities


def main():

    #######dataset path
    #datadir = sys.argv[1]
    #pathDataset1 = datadir+'/humanData.txt'

    #data = loadData('data_sets_clustering/humanData.txt')

  #  print data
    #print error
    #purities = []

   # for i in range(1, 7):
   #     centroids, clusters, error = kmeans(data, i, maxIter=1000)
   #     purities.append(pur)

    #centroids, clusters, error = kmeans(data, 2, maxIter=1000)  #
    #pur = purity(data, clusters)

    #kneeFinding(data,range(1,7))

    #visualizeClusters(data, clusters)

    datadir = sys.argv[1]
    pathDataset1 = datadir + '/humanData.txt'
    dataset1 = loadData(pathDataset1)

    # Q4
    kneeFinding(dataset1, range(1, 7))

    # Q5
    centroids, clusters, error = kmeans(dataset1, 2, maxIter=1000)
    purity(dataset1, clusters)

if __name__ == "__main__":
    main()