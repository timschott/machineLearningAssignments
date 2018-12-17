#!/usr/bin/python

import sys
import math
import os
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
###############################################################################

def transfer(fileDj, vocabulary, label):
    # print ('transferring ' + fileDj)

    # stem love
    # count freqs
    # return BOW row, and a label Row.
    f = open(fileDj, 'r')
    #    print(data)
    lines = f.read().splitlines()
    #if (fileDj == 'data_sets/training_set/neg/cv465_23401.txt'):

    lines = [l.replace('loving', 'love') for l in lines]
    lines = [l.replace('loved', 'love') for l in lines]
    lines = [l.replace('loves', 'love') for l in lines]

    #  count freqs.
    # need to look at the words, now.

    words = []

    words.append([l.split() for l in lines])
    words_list = [item for sublist in words for item in sublist]
    words = [item for sublist in words_list for item in sublist]

    vocabulary_dict = {el: 0 for el in vocabulary}

    for word in words:
        if word in vocabulary:
            vocabulary_dict[word] += 1
        # i believe that we are supposed to label all untracked words as UNK.
        else:
            vocabulary_dict['UNK'] +=1


    # vocabulary_dict is now what we need to build out our bag of words with.
    # make the order is consistent, then put in a list,
    # the filename, and the counts.

    list_of_tuples = [(key, vocabulary_dict[key]) for key in vocabulary]

    vocabulary_ordered_dict = OrderedDict(list_of_tuples)

    x_list = []

    fileName = fileDj.split('/')[3].split('.')[0]

    x_list.append(fileName)

    for key in vocabulary_ordered_dict:
        x_list.append(vocabulary_ordered_dict[key])

    y_list = []
    y_list.append(fileName)
    y_list.append(label)

    return x_list, y_list

    #return BOWDj


def loadData(Path):
    vocab = ['love', 'wonderful', 'best', 'great', 'superb',
                 'still', 'beautiful', 'bad', 'worst', 'stupid',
                 'waste', 'boring', '?', '!', 'UNK']

    # positive training.

    pos_train_dir = Path + "training_set/pos/"

    x_pos_train = []
    y_pos_train = []

    for filename in os.listdir(pos_train_dir):
        filename = pos_train_dir + filename
        x_pos_train_temp, y_pos_train_tmp = transfer(filename, vocab, 'pos')
        x_pos_train.append(x_pos_train_temp)
        y_pos_train.append(y_pos_train_tmp)

    # negative training

    x_neg_train = []
    y_neg_train = []

    neg_train_dir = Path + "training_set/neg/"

    for filename in os.listdir(neg_train_dir):
        filename = neg_train_dir + filename
        x_neg_train_temp, y_neg_train_temp = transfer(filename, vocab, 'neg')
        x_neg_train.append(x_neg_train_temp)
        y_neg_train.append(y_neg_train_temp)

    # positive testing

    x_pos_test = []
    y_pos_test = []

    neg_train_dir = Path + "test_set/pos/"

    for filename in os.listdir(neg_train_dir):
        filename = neg_train_dir + filename
        x_pos_test_temp, y_pos_test_temp = transfer(filename, vocab, 'pos')
        x_pos_test.append(x_pos_test_temp)
        y_pos_test.append(y_pos_test_temp)

    # negative testing

    x_neg_test = []
    y_neg_test = []

    neg_train_dir = Path + "test_set/neg/"

    for filename in os.listdir(neg_train_dir):
        filename = neg_train_dir + filename
        x_neg_test_temp, y_neg_test_temp = transfer(filename, vocab, 'neg')
        x_neg_test.append(x_neg_test_temp)
        y_neg_test.append(y_neg_test_temp)

    # make a big giant thing with x and y
    # "mega"
    # giant time..

    x_pos_train = np.array(x_pos_train)
    x_neg_train = np.array(x_neg_train)

    y_pos_train = np.array(y_pos_train)
    y_neg_train = np.array(y_neg_train)

    x_train = np.concatenate((x_pos_train, x_neg_train), axis=0)
    y_train = np.concatenate((y_pos_train, y_neg_train), axis=0)

    x_pos_test = np.array(x_pos_test)
    x_neg_test = np.array(x_neg_test)

    y_pos_test = np.array(y_pos_test)
    y_neg_test = np.array(y_neg_test)

    x_test = np.concatenate((x_pos_test, x_neg_test), axis=0)
    y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)

    return x_train, x_test, y_train, y_test


def loadTestData(TestPath):
    vocab = ['love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful', 'bad', 'worst', 'stupid', 'waste',
             'boring', '?', '!', 'UNK']

    x_pos_test = []
    y_pos_test = []

    neg_train_dir = TestPath + "/pos/"

    for filename in os.listdir(neg_train_dir):
        filename = neg_train_dir + filename
        x_pos_test_temp, y_pos_test_temp = transfer(filename, vocab, 'pos')
        x_pos_test.append(x_pos_test_temp)
        y_pos_test.append(y_pos_test_temp)

    # negative testing

    x_neg_test = []
    y_neg_test = []

    neg_train_dir = TestPath + "/neg/"

    for filename in os.listdir(neg_train_dir):
        filename = neg_train_dir + filename
        x_neg_test_temp, y_neg_test_temp = transfer(filename, vocab, 'neg')
        x_neg_test.append(x_neg_test_temp)
        y_neg_test.append(y_neg_test_temp)

    # make a big giant thing with x and y
    # "mega"
    # giant time..

    x_pos_test = np.array(x_pos_test)
    x_neg_test = np.array(x_neg_test)

    y_pos_test = np.array(y_pos_test)
    y_neg_test = np.array(y_neg_test)

    x_test = np.concatenate((x_pos_test, x_neg_test), axis=0)
    y_test = np.concatenate((y_pos_test, y_neg_test), axis=0)

    return x_test, y_test


def naiveBayesMulFeature_train(Xtrain, ytrain):

    # MAKE "mega doc" of all of our x_train_pos and x_train_neg

    x_train_pos = Xtrain[0:700,1:]

    x_train_pos = x_train_pos.astype(np.float)

    x_train_neg = Xtrain[700:1400,1:]

    x_train_neg = x_train_neg.astype(np.float)

    # mega doc...
    # ...... total + of words is the sum of all rows.
    # ...... total # of hits for wk is found by
    # checking that row, adding 1,
    # dividing by sum of row + 15

    # total # of words in each mega doc.
    # sum all elements
    pos_sum = sum(x_train_pos.sum(axis=1))
    neg_sum = sum(x_train_neg.sum(axis=1))

    # frequency of each character in mega doc.
    # sum the columns

    pos_col_sums = x_train_pos.sum(axis=0)
    neg_col_sums = x_train_neg.sum(axis=0)

    # containers

    thetaPos = []
    thetaNeg = []
    #print(pos_col_sums)


    for freq in pos_col_sums:
        #print freq
        temp = (((freq + 1)) / (pos_sum + 15))
        value = math.log(.5) + math.log(temp)
        thetaPos.append(value)

    for freq in neg_col_sums:
        # print freq
    #    print(((freq + 1)) / (neg_sum + 15)) * 100
        temp = (((freq + 1)) / (neg_sum + 15))
        value = math.log(.5) + math.log(temp)
        thetaNeg.append(value)

    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(Xtest, ytest,thetaPos, thetaNeg):

    x_test = Xtest[0:600, 1:]

    x_test = x_test.astype(np.float)

    #print pos_col_sums
    # calculate predictions

    # predictions are . 5 + weights ^ number of occurences.
    # frequency of each character in testing docs.

    pos_results = []
    neg_results = []

    for i in range(0, len(x_test)):
        pos_test = 0
        neg_test = 0
        for j in range(0, 15):
            pos_weight = thetaPos[j]
            neg_weight = thetaNeg[j]
            pos_test += ((pos_weight) * x_test[i,j])
            neg_test += ((neg_weight) * x_test[i,j])
        pos_results.append(pos_test)
        neg_results.append(neg_test)

    yPredict = []

    for i in range(0, len(pos_results)):
        if(pos_results[i] >= neg_results[i]):
            yPredict.append('pos')
        else:
            yPredict.append('neg')

    true_y_values = ytest[:, 1]

   # print yPredict
   # print true_y_values

    accurate = []
    for i in range(0, len(yPredict)):
        if yPredict[i] == true_y_values[i]:
            accurate.append(i)

    Accuracy = len(accurate)/600.0

    return yPredict, Accuracy

def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):

    x_train = Xtrain[0:1400, 1:]

    x_train = x_train.astype(np.float)

    x_test = Xtest[0:600, 1:]

    x_test = x_test.astype(np.float)

    y_train = ytrain[0:1400, 1:]

    y_train = y_train.ravel()

    y_train_new = []

    for i in range(0, len(y_train)):
        if y_train[i] == 'pos':
            y_train_new.append('1')
        else:
            y_train_new.append('0')

    y_train = [float(i) for i in y_train_new]

    y_test = ytest[0:1400, 1:]

    y_test = y_test.ravel()

    y_test_new = []

    for i in range(0, len(y_test)):
        if y_test[i] == 'pos':
            y_test_new.append('1')
        else:
            y_test_new.append('0')

    y_test = [float(i) for i in y_test_new]

    nb = MultinomialNB()

    nb.fit(x_train, y_train)

    predicted = nb.predict(x_test)
    Accuracy = metrics.accuracy_score(y_test, predicted)
    return Accuracy


def naiveBayesMulFeature_testDirectOne(path, thetaPos, thetaNeg):
    # i think this is just supposed to call loadData?
    return 0
#   return yPredict


def naiveBayesMulFeature_testDirect(path, thetaPos, thetaNeg):
    # i think this is just supposed to call loadData?

    x_test, y_test = loadTestData(path)

    yPredict, Accuracy = naiveBayesMulFeature_test(x_test, y_test, thetaPos, thetaNeg)

    return yPredict, Accuracy


def naiveBayesBernFeature_train(Xtrain, ytrain):

    # okay for Bernoullil we just need to change this matrix into 1's and 0's

    x_train_pos = Xtrain[0:700, 1:]

    x_train_pos = x_train_pos.astype(np.float)

    x_train_neg = Xtrain[700:1400, 1:]

    x_train_neg = x_train_neg.astype(np.float)

    # mega doc...
    # ...... total + of words is the sum of all rows.
    # ...... total # of hits for wk is found by
    # checking that row, adding 1,
    # dividing by sum of row + 15

    # total # of words in each mega doc.
    # sum all elements
    pos_sum = sum(x_train_pos.sum(axis=1))
    neg_sum = sum(x_train_neg.sum(axis=1))


    for i in range(0, len(x_train_pos)):
        for j in range(0, 15):
            if x_train_pos[i,j] != 0:
                x_train_pos[i,j] = 1

    for i in range(0, len(x_train_neg)):
        for j in range(0, 15):
            if x_train_neg[i, j] != 0:
                x_train_neg[i, j] = 1

    pos_true_results = []
    pos_false_results = []

    neg_true_results = []
    neg_false_results = []

    pos_col_sums = x_train_pos.sum(axis=0)
    neg_col_sums = x_train_neg.sum(axis=0)

    #print pos_col_sums

    for freq in pos_col_sums:
        temp = (freq + 1) / (700 + 2)
        #value = math.log(.5) + math.log(temp)
        pos_true_results.append(temp)

    for prob in pos_true_results:
        pos_false_results.append(1 - prob)

    for freq in neg_col_sums:
        temp = (freq + 1) / (700 + 2)
        #value = math.log(.5) + math.log(temp)
        neg_true_results.append(temp)

    for prob in neg_true_results:
        neg_false_results.append(1 - prob)

    thetaPosTrue = pos_true_results
    thetaNegTrue = neg_true_results

    return thetaPosTrue, thetaNegTrue

    
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):

    # steps:
    # convert Xtest into 1s and 0s
    # remove name

    x_test = Xtest[0:600, 1:]
    x_test = x_test.astype(np.float)

    # convert to 1s and 0s


    for i in range(0, len(x_test)):
        for j in range(0, 15):
            if x_test[i, j] != 0:
                x_test[i, j] = 1.0

    # moving on.
    #


    y_test = ytest[0:600, 1:]

    pos_results = []
    neg_results = []

    for i in range(0, len(x_test)):
        pos_temp = 0
        neg_temp = 0
        for j in range(0, 15):
            if x_test[i,j] == 1.0:
                pos_temp += math.log(thetaPosTrue[j])
                neg_temp += math.log(thetaNegTrue[j])
            else:
                pos_temp += (1- math.log(thetaPosTrue[j]))
                neg_temp += (1- math.log(thetaNegTrue[j]))
        pos_results.append(pos_temp)
        neg_results.append(neg_temp)

    yPredict = []

    for i in range(0, len(pos_results)):
        if (pos_results[i] >= neg_results[i]):
            yPredict.append('pos')
        else:
            yPredict.append('neg')

    true_y_values = ytest[:, 1]

    accurate = []
    for i in range(0, len(yPredict)):
        if yPredict[i] == true_y_values[i]:
            accurate.append(i)

    Accuracy = len(accurate) / 600.0
    return yPredict, Accuracy


if __name__ == "__main__":
   # if len(sys.argv) != 3:
   #     print "Usage: python naiveBayes.py dataSetPath testSetPath"
    #    sys.exit()

    print "--------------------"
    #textDataSetsDirectoryFullPath = sys.argv[1]
    #testFileDirectoryFullPath = sys.argv[2]

    textDataSetsDirectoryFullPath = 'data_sets/'
    testFileDirectoryFullPath = 'data_sets/test_set'
    np.set_printoptions(suppress=True)

    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)
    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)

    print "thetaPos = ", thetaPos
    print "thetaNeg = ", thetaNeg

    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print ('MNBC classification accuracy = ' + str(Accuracy * 100) + '%')

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print ('Sklearn MultinomialNB accuracy = ' + str(Accuracy * 100) + '%')

    yPredict, Direct_Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg)
    print ('Directly MNBC testing accuracy = ' + str(Direct_Accuracy * 100) + '%')
    print "--------------------"

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print "thetaPosTrue = ", thetaPosTrue
    print "thetaNegTrue = ", thetaNegTrue
    print "--------------------"
    naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print ('BNBC classification accuracy = ' + str(Accuracy * 100) + '%')
    print "--------------------"