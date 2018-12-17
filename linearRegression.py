# Machine Learning HW1
# tim schott tcs9pk

import numpy as np
import csv
import matplotlib.pyplot as plt

# import data set from file
def loadDataSet(filename):
    dummy = []
    xValues = []
    yValues = []

#read in file
    with open(filename, 'r') as data:
        lines = csv.reader(data, delimiter='\t')
        for dummies, xVals, yVals in lines:
            #create lists to store bias, x and y
            dummy.append(dummies)
            xValues.append(xVals)
            yValues.append(yVals)

    #make floats not string
    dummy = [float(i) for i in dummy]
    xValues = [float(i) for i in xValues]
    yValues = [float(i) for i in yValues]

    return dummy, xValues, yValues

#standard regression using the normal equation
def standRegresOpt1(dumbList, xList, yList):

    n = len(xList)
    #np reshape so we can take advantage of np matrix operations
    dummiesToAdd = np.reshape(dumbList, (n, 1))

    #append bias to x matrix
    xMat = np.reshape(xList, (n, 1))
    xMat = np.append(dummiesToAdd, xMat, axis=1)

    yMat = np.reshape(yList, (n, 1))

    xMat_t = xMat.T
    #boil down to our intended theta
    theta = np.linalg.inv(xMat_t.dot(xMat))
    theta = theta.dot(xMat_t)
    theta = theta.dot(yMat)

    print 'Linear Regression theta: ', theta
    intercept = theta[0]
    slope = theta[1]
    #create regression line
    line = slope * xList + intercept

    plt.plot(xList, yList, 'or')
    plt.plot(xList, line)
    plt.title('Normal equation regression results')
    #plt.savefig('normalEq.png')
    plt.show()

    return theta

# gradient algorithm
def gradientAlgorithm(b, m, xList, yList, alpha):
    bGrad = 0
    mGrad = 0
    # carry out calculations for every point in the data set
    for i in range(0, 200):
        x = xList[i]
        y = yList[i]
        bGrad += (-2/200.0) * (y-(m*x+b))
        mGrad += (-2/200.0) * (x) * (y-(m*x+b))
    #increment our coefficients
    bNext = b - (alpha * bGrad)
    mNext = m - (alpha * mGrad)

    return [bNext, mNext]

# mean squared error calculation
def errorCalc(b, m, xList, yList):
    error = 0
    for i in range(0, 200):
        x = xList[i]
        y = yList[i]
        error += (y-(b*x+m)) ** 2

    error = error/200

    return error

# epoch handler
def standRegresOpt2(xList, yList, starting_b, starting_m, alpha, epochs):
    b = starting_b
    m = starting_m
    epochList = []
    errorList = []
    for i in range(epochs):
        # keep track of how we did
        b, m = gradientAlgorithm(b, m, xList, yList, alpha)
        epochList.append(i)
        errorList.append(errorCalc(b, m, xList, yList))
    theta = b, m

    plt.plot(epochList, errorList, 'or')
    plt.title('Epoch number and error output')
    #plt.savefig('error.png')
    plt.show()

    return theta, epochList, errorList

#plot the gradient work

def plotGradientWork(xVals, yVals, theta):

    intercept = theta[0]
    slope = theta[1]

    newVal = []
    for el in xVals:
        newVal.append(el * slope + intercept)

    plt.plot(xVals, yVals, 'or')
    plt.plot(xVals, newVal)
    plt.title('Standard Gradient Descent work')
    #plt.savefig('sgdPlot.png')
    plt.show()

#main method to generate plots and thetas
def main():
    dumbList, xList, yList = loadDataSet("Q2data.txt")
    plt.plot(xList, yList, 'or')
    plt.title('Data View')
    #plt.savefig('justXAndY.png')
    standRegresOpt1(dumbList, xList, yList)
    gdTheta, epochList, errorList = standRegresOpt2(xList, yList, 1, 1, .001, 10000)
    print 'GD theta: ', gdTheta
    plotGradientWork(xList, yList, gdTheta)

if __name__ == "__main__":
    main()
