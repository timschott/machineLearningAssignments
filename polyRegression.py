# Machine Learning HW2
# First Programming Task: Polynomial Regression

__author__ = 'Timothy Schott tcs9pk'

import numpy as np
import csv
import matplotlib.pyplot as plt

# more imports

def loadDataSet(filename):
    dummy = []
    xValues = []
    yValues = []

    # read in file
    with open(filename, 'r') as data:
        lines = csv.reader(data, delimiter='\t')
        for dummies, xVals, yVals in lines:
            # create lists to store bias, x and y
            dummy.append(dummies)
            xValues.append(xVals)
            yValues.append(yVals)

    # make floats not string
    dummy = [float(i) for i in dummy]
    xValues = [float(i) for i in xValues]
    yValues = [float(i) for i in yValues]

    return dummy, xValues, yValues

def errorCalc(theta, d, xMat, yList):

    error = 0
    n = len(yList)

    if d != 0:
        for i in range(1, d + 1):

            currentColumn = xMat[:, i]
            currentTheta = theta[i]

            for j in range(0, 10):
                error += np.power(np.array(yList[j])-np.array(np.multiply(currentColumn[j],currentTheta)), 2)
            error = error / 200
    else:
        for x in range(0, n):
            error += np.power(np.array(yList[x]) - np.array(theta[0]), 2)
        error = error/200

    return error

def polyRegresTrain(X_train, Y_train, dumbList, d):

    """Given the training data, learn a polynomial regression
    model of degree d.
    Output: a dx1 vector containing the learned coefficients for the regression model.
    I.e., y_predicted = theta0 + theta1 * x + theta2 * x^2 + ... + thetad * x^d
    """

    n = len(X_train)
    # np reshape so we can take advantage of np matrix operations

    dummiesToAdd = np.reshape(dumbList, (n, 1))

    # append bias to x matrix

    xMat = np.reshape(X_train, (n, 1))

    xMat = np.append(dummiesToAdd, xMat, axis=1)
    # grow x matrix based on the degree d

    if d > 1:
        for i in range(2, d+1):
            values = np.power(X_train, i)
            valuesCol = np.reshape(values, (n, 1))
            #print(valuesCol)
            xMat = np.append(xMat, valuesCol, axis=1)
            #print np.shape(xMat)
    if d == 0:
        xMat = dummiesToAdd

    # theta = inv(X ^ T * X) * X ^ T * y

    #print('xmat,', np.shape(xMat))

    yMat = np.reshape(Y_train, (n, 1))
    #print('ymat', np.shape(yMat))

    xMat_t = np.transpose(xMat)
    #print('xmat - t', np.shape(xMat_t))
    #print('dot', np.shape(np.matmul(xMat_t, xMat)))

    tg = np.linalg.inv(np.matmul(xMat_t, xMat))
    #print('inv', np.shape(np.linalg.inv(tg)))

    ## INVERSE is 200 by 200. okay. now we need to dot that with xMat_t and yMat
    #print('last bit', np.shape(np.matmul(xMat_t, yMat)))
    lastBit = np.matmul(xMat_t, yMat)

    #print('inversed', np.linalg.inv(xMat_t).dot(xMat))
    theta = np.matmul(tg, lastBit)
    #print np.shape(theta)
    # boil down to our intended theta

    #print('theta', theta, 'degrees', d)

    sum = 0
    for i in Y_train:
        sum += i

    theta[0] = sum/n

    #for j in theta:
    #    print 'theta ', j

    if d == 0:
         theta = theta[0]

    #print(np.shape(theta))
    polyTitle = 'Normal equation results with d = ' + str(d)

    # hand off structs to error calculator:

    mse = errorCalc(theta, d, xMat, yMat)
    #print 'Poly regression train MSE: ', mse
    return theta, mse

    # create a distribution of random data

def polyRegresTrainPlot(theta, d):

    x_bin = np.linspace(0, 7, 1000)  # uniformly sample the range (0, 7) 1000 times

    pred_curve = []

    # apparently have to go the matrix route :-)

    xRandMat = np.reshape(x_bin, (1000, 1))
    dummiesForRand = np.repeat(1.0, 1000)
    dummiesMat = np.reshape(dummiesForRand, (1000, 1))

    xRandMat = np.column_stack((xRandMat, dummiesForRand))
    xRandMat[:, [0, 1]] = xRandMat[:, [1, 0]]

    if d > 1:
        for i in range(2, d+1):
            val = np.power(x_bin, i)
            valCol = np.reshape(val, (1000, 1))
            xRandMat = np.append(xRandMat, valCol, axis=1)

    if d == 0:
        xRandMat = dummiesForRand

    #initialize the curve.
    pred_curve = theta[0]

    for i in range(1, d + 1):
        param = theta[i] * xRandMat[:, i]
        pred_curve = pred_curve + param

    if d == 0:
        pred_curve = np.repeat(theta[0], 1000)

    fileTitle = 'outputwith'+ str(d)+'.png'
    plt.plot(x_bin, pred_curve)
    plt.title('normal Equation with degree' + str(d))
    #plt.savefig(fileTitle)
    plt.show()

    return 0


def polyRegresTest(X_test, Y_test, dumb_list, degree, ptheta):

    n = len(X_test)
    dumb_list = dumb_list[:100]

    # np reshape so we can take advantage of np matrix operations

    dummiesToAdd = np.reshape(dumb_list, (n, 1))

    # append bias to x matrix

    xMat = np.reshape(X_test, (n, 1))
    xMat = np.append(dummiesToAdd, xMat, axis=1)

    # grow x matrix based on the degree d
    if degree > 1:
        for i in range(2, degree+1):
            values = np.power(X_test, i)
            valuesCol = np.reshape(values, (n, 1))
            xMat = np.append(xMat, valuesCol, axis=1)


    yMat = np.reshape(Y_test, (n, 1))

    MSETestLoss = errorCalc(ptheta, degree, xMat, yMat)

    #print 'Poly regression test MSE: ', MSETestLoss

    return MSETestLoss


def trainAndValidate(X_train, Y_train, X_test, Y_test, dumb_list, degree):
    """Iteratively call polyRegresTrain and polyRegresTest to find the best polynomial
    model for the dataset.
    ie we need to figure out what d is.
    Display the following plots:
        (1) Training MSE vs degree
        (2) Testing MSE vs degree
    Return pthetaBest, a vector containing the coefficients for the best polynomial model
    """

    TestErrorList = []
    TrainErrorList = []
    ErrorMat = np.zeros(shape=(degree+1, 2))

    DiffList = []
    for i in range(0, degree+1):
        testTheta, testMse = polyRegresTrain(X_train, Y_train, dumb_list, i)
        TestErrorList.append(testMse)
        trainMse = polyRegresTest(X_test, Y_test, dumb_list, i, testTheta)
        TrainErrorList.append(trainMse)

    TestError = np.reshape(TestErrorList, (degree+1, 1))
    TrainError = np.reshape(TrainErrorList, (degree+1, 1))

    ErrorMat = np.append(TestError, ErrorMat, axis=1)
    ErrorMat = np.append(TrainError, ErrorMat, axis=1)

    ErrorMat = ErrorMat[:, [0, 1, 2]]
    ErrorMat[:, 2] = abs(ErrorMat[:, 0] - ErrorMat[:, 1])

    # min error is when we have 4 degrees of freedom.

    ptheta = polyRegresTrain(X_train, Y_train, dumb_list, 4)[:-1]
    finalPlots(TrainError, TestError, degree, ptheta)

    return ptheta

def finalPlots(trainLoss, testLoss, degree, ptheta):

    # degree, theta
    deg = list(range(0,9))
    plt.plot(deg, testLoss)
    plt.title('Test Loss vs Degree')
    fileTitle = 'testLoss'
    plt.savefig('testLoss.png')
    plt.show()

    plt.plot(deg, trainLoss)
    plt.title('Training Loss vs Degree')
    plt.savefig('trainingLoss.png')
    plt.show()

    #plt.title('Final plot: degree is 4.')
    ptheta = np.asarray(ptheta).reshape(-1)
    polyRegresTrainPlot(ptheta, 4)
    #plt.show()

    return 0

if __name__ == "__main__":

    degree = 8
    dummy, X_train, Y_train = loadDataSet('polyRegress_train.txt')
    ptheta, MSETrainLoss = polyRegresTrain(X_train, Y_train, dummy, degree)

    polyRegresTrainPlot(ptheta, degree)

    ### Function 2: evaluate the polynomial model on the validation set
    dummy_test, X_test, Y_test = loadDataSet('polyRegress_validation.txt')
    MSETestLoss = polyRegresTest(X_test, Y_test, dummy_test, degree, ptheta)

    ### Function 3: use the previous two methods to find a good regression model
    pthetaBest = trainAndValidate(X_train, Y_train, X_test, Y_test, dummy, degree)

    for i in range(0, 9):
        dummy, X_train, Y_train = loadDataSet('polyRegress_train.txt')
        ptheta, MSETrainLoss = polyRegresTrain(X_train, Y_train, dummy, i)
        print 'degree ', i, 'theta is ', ptheta