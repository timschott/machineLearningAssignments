# Machine Learning HW2-Ridge

__author__ = 'Tim Schott tcs9pk'

import csv
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
# more imports 


def loadDataSet(filename):
    #your code

    dummy = []
    x1Values = []
    x2Values = []
    yValues = []

    # read in file
    with open(filename, 'r') as data:
        lines = csv.reader(data, delimiter=' ')
        for dummies, xVals1, xVals2, yVals in lines:
            # create lists to store bias, x and y
            dummy.append(dummies)
            x1Values.append(xVals1)
            x2Values.append(xVals2)
            yValues.append(yVals)

    # make floats not string
    dummy = [float(i) for i in dummy]
    x1Values = [float(i) for i in x1Values]
    x2Values = [float(i) for i in x2Values]
    yValues = [float(i) for i in yValues]

    n = len(dummy)

    dummiesToAdd = np.reshape(dummy, (n,1))

    xMat = np.reshape(x2Values, (n, 1))

    xValuesNeedToAdd = np.reshape(x1Values, (n, 1))

    xMat = np.append(xValuesNeedToAdd, xMat, axis=1)

    xMat = np.append(dummiesToAdd, xMat, axis=1)

    yMat = np.reshape(yValues, (n, 1))

    return xMat, yMat

def errorCalc(xTest, yVal, beta):

    #X * theta - yActual
    predMat = np.dot(xTest, beta)
    residuals = np.subtract(predMat, yVal)
    ssq = np.sum(residuals**2)
    ssq = ssq/50

    return ssq

def ridgeRegress(xVal, yVal, lambdaV, showFigure=True):

    #Beta Star = (X_t dot X + Lambda * Identity) inv.. dot (X_t dot y)

    xMat_t = np.transpose(xVal)

    xResultMat = np.matmul(xMat_t, xVal)

    lMat = lambdaV * np.identity(3)

    #print(lMat)
    #print(shape(lMat)

    plus = np.add(xResultMat, lMat)
    #print(np.shape(xResultMat))
    invQuant = np.linalg.inv(plus)
    #print(np.shape(invQuant))
    xMat_tDotY = np.matmul(xMat_t, yVal)
    #print(np.shape(xMat_tDotY))

    beta = np.matmul(invQuant, xMat_tDotY)
    #print(np.shape(beta))
    return beta


def cv(xVal, yVal):
     # start with the set of lambdas from 0 to 1 while increasing every 2

    lambdaList = (np.linspace(0, 1, num=51))

    # need to shuffle our data

    np.random.seed(37)
    np.random.shuffle(xVal)
    np.random.seed(37)
    np.random.shuffle(yVal)

    # Fold 1
    training1 = xVal[:150, :]
    training1YValues = yVal[:150,]
    testing1 = xVal[150:200, ]
    testing1YValues = yVal[150:200,]

    # Fold 2
    training2 = xVal[50:200, :]
    training2YValues = yVal[50:200]
    testing2 = xVal[0:50, ]
    testing2YValues = yVal[0:50,]

    # Fold 3
    # x training
    training3 = xVal[100:200, :]
    xValuesToAddTo3 = xVal[0:50, :]
    training3 = np.append(xValuesToAddTo3, training3, axis=0)

    # y training
    training3YValues = yVal[100:200, ]
    yValuesToAddTo3 = yVal[0:50, :]
    training3YValues = np.append(yValuesToAddTo3, training3YValues, axis=0)

    # x testing
    testing3 = xVal[50:100, ]

    # y testing
    testing3YValues= yVal[50:100,]

    # Fold 4

    # X training

    training4 = xVal[150:200, ]
    xValuesToAddTo4 = xVal[0:100, :]
    training4 = np.append(xValuesToAddTo4, training4, axis=0)

    # Y training
    training4YValues = yVal[150:200, ]
    yValuesToAddTo4 = yVal[0:100, :]
    training4YValues = np.append(yValuesToAddTo4, training4YValues, axis=0)

    # X Testing
    testing4 = xVal[100:150,]

    # Y Testing

    testing4YValues = yVal[100:150]

    showFigure = False

    ErrorList = []

    best = 10000

    #print(len(lambdaList))
    #print(lambdaList)
    for i in lambdaList:
        #print(' lambda is, ', i)
        beta1 = ridgeRegress(training1, training1YValues, i, showFigure)
        beta2 = ridgeRegress(training2, training2YValues, i, showFigure)
        beta3 = ridgeRegress(training3, training3YValues, i, showFigure)
        beta4 = ridgeRegress(training4, training4YValues, i, showFigure)
        #print(' beta1, ', beta1)

        error1 = errorCalc(testing1, testing1YValues, beta1)
        error2 = errorCalc(testing2, testing2YValues, beta2)
        error3 = errorCalc(testing3, testing3YValues, beta3)
        #print(' error 3 ', error3)
        error4 = errorCalc(testing4, testing4YValues, beta4)
        #print(' errror2, ', error2)
        #print(' errror3, ', error3)
        #print(' errror4, ', error4)

        value = ((error1+error2+error3+error4)/4)
        #print value
        ErrorList.append(value)
        temp = min(ErrorList)

        if value <= temp:
            best = i


    ## Plot each item in error list against the correct lambda

    #print(ErrorList)
    plt.title('Error and Lambda')
    plt.plot(lambdaList, ErrorList)
    plt.savefig('Error and Lambda.png')
    plt.show()

    #print('smallest in error list,' , min(ErrorList))
    #print('best', best)

    return best

def standRegress(xMat, yMat):
     # use your standRegress code from HW1  and show figure

    xMat_t = np.transpose(xMat)

    tg = np.linalg.inv(np.matmul(xMat_t, xMat))

    lastBit = np.matmul(xMat_t, yMat)

    theta = np.matmul(tg, lastBit)

    return theta

def planePlot(xVal, yVal, beta, title, fileName):
    x1 = np.multiply(xVal[:, 1], beta[1])
    x2 = np.multiply(xVal[:, 2], beta[2])
    y = yVal + beta[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x1, x2, y, color='b')
    plt.title(title)
    plt.savefig(fileName)
    plt.show()

    return 0
if __name__ == "__main__":
    xVal, yVal = loadDataSet('RRdata.txt')
    #print(len(xVal))
    #np.random.seed(37)
    #np.random.shuffle(xVal)
    #np.random.seed(37)
    #np.random.shuffle(yVal)
    ## Assuming an 80-20 training / testing split ?
    betaLR = ridgeRegress(xVal[1:200], yVal[1:200], lambdaV=0)
    print 'lambda is 0, ', betaLR
    title = 'First Regression, Lambda is 0.'
    planePlot(xVal, yVal, betaLR, title, 'lambda0plane.png')
    xVal, yVal = loadDataSet('RRdata.txt')
    lambdaBest = cv(xVal, yVal)
    # lambdaBest
    betaRR = ridgeRegress(xVal, yVal, lambdaV=lambdaBest)
    print 'lambda is 0.5 ', betaRR
    title = 'Ridge Regression, Lambda is 0.5'
    planePlot(xVal, yVal, betaRR, title, 'lambda0.5plane.png')
    #planePlot(xVal, yVal, betaRR)

    # depending on the data structure you use for xVal and yVal, the following line may need some change
    #standRegress(xVal[:,:2], yVal[])
