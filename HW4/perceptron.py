import argparse
import numpy as np
import pandas as pd
import time

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}
        # TODO implement this
        self.w = np.random.uniform(size = (len(xFeat[0])))


        # Follows basic perception algorithm discussed in class
        for epoch in range(self.mEpoch):
            yHat = np.zeros((len(y)))
            for index,row in enumerate(xFeat):
                if self.w.dot(row) >= 0:
                    yHat[index] = 1
                else:
                    yHat[index] = 0
                

                if yHat[index]  < y[index]:
                    self.w += row
                elif yHat[index]  > y[index]:
                    self.w -= row

            err = calc_mistakes(yHat,y)        
            stats[epoch] = err
            if err == 0:
                break
            

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """


        yHat = []
        for index,row in enumerate(xFeat):
                if self.w.dot(row) >= 0:
                    yHat.append(1)
                else:
                    yHat.append(0)

        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    err = 0
    for i in range(len(yTrue)):
        if yHat[i] != yTrue[i]:
            err += 1 
    return err


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))

    # Code to find optimal Epoch
    # finalEpoch = 0
    # finalNumMistake = 99999999999999
    # for epoch in range(args.epoch):
    #     np.random.seed(args.seed)   
    #     model = Perceptron(epoch)
    #     trainStats = model.train(xTrain, yTrain)
    #     # print(trainStats)
    #     yHat = model.predict(xTest)
    #     # print out the number of mistakes
    #     print("---------------------------------------")
    #     print("Epoch Number")
    #     print(epoch)
    #     print("Number of mistakes on the test dataset")
    #     err = calc_mistakes(yHat, yTest)
    #     print(err)
    #     if err < finalNumMistake:
    #         finalEpoch = epoch
    #         finalNumMistake = err

    # print("----------------")
    # print("Best Epoch")
    # print(finalEpoch)
    # print("Lowest Mistake")
    # print(finalNumMistake)

    # Code to find 15 most positive and negative words
    # colNames = pd.read_csv(args.xTrain)
    # colNames = list(colNames.columns)
    # indexesBig = np.argpartition(model.w,-15)[-15:]
    # indexesSmall = model.w.argsort()[:15]


    # colNamesBig = [colNames[i] for i in indexesBig]
    # colNamesSmall = [colNames[i] for i in indexesSmall]
    # print(colNamesBig)
    # print(colNamesSmall)
if __name__ == "__main__":
    main()