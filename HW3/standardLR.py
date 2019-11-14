import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SOMETHING

        data = {'time':0,'train-mse':0,'test-mse':0}

        timeE = time.time()

        # Calculates the closed form solution 
        # The code belows pretty much directly transfers the math equation into code
        xTrainTrans = xTrain.transpose()

        temp = np.dot(xTrainTrans, xTrain)
        tempInv = np.linalg.pinv(temp)
        
        betaTemp = np.dot(tempInv,xTrainTrans)

        betaTemp = np.dot(betaTemp,yTrain)

        LinearRegression.beta = betaTemp

        data['train-mse'] = LinearRegression.mse(self,xTrain,yTrain)

        xTestTrans = xTest.transpose()

        tempTest = np.dot(xTestTrans,xTest)
        tempInv = np.linalg.pinv(tempTest)

        betaTemp = np.dot(tempInv,xTestTrans)

        betaTemp = np.dot(betaTemp,yTest)

        LinearRegression.beta = betaTemp

        data['test-mse'] = LinearRegression.mse(self,xTest,yTest)

        timeE = time.time() - timeE

        data['time'] = timeE

        trainStats = {0:data}

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
