import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Knn(object):
    k = 0    # number of neighbors to use
      # Array of labels associated with training data
    xFeat = []
    y = []
    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need

        ## Stores the training data to be used late
        self.xFeat = pd.DataFrame(xFeat)
        self.y = y
        return self

    def findNeighbor(self,row, yHat):

        nearNeig = []
        eucDist = []
        rowIndex = 0

        ## Calculate the Eulcidean dist between the trainging data and specific point
        for val in (self.xFeat).values:
            dist = np.linalg.norm(row-val)
            eucDist.append(dist)

        indexArr = np.argsort(eucDist)
        
        ## Adds all the nearest neighbors to an array
        for i in range(self.k):
            nearNeig.append(indexArr[i])
        

        count = 0

        ## Counts the results and determine whether the prediction is 1 or 0
        for neigh in nearNeig:
            if self.y.values[neigh] > 0:
                count += 1
        if count / float(len(nearNeig)) >= .5:
            yHat.append(1)
        else:
            yHat.append(0)

        


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
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO

        ## Iterates through the data and finds the nearest neighbor
        xFeat = pd.DataFrame(xFeat)
        for row in xFeat.values:
            self.findNeighbor(row,yHat)
        
        return yHat


    

def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0

    ## Basic method to calculate accuracy
    for i in range(len(yHat)):
        if yHat[i] == yTrue[i]:
            acc += 1
    acc = acc / float(len(yTrue))
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])    


    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


## Code I used to Graph K values and Accuracy
    # knnStor = []
    # trAcc = []
    # teAcc = []
    # kVal = []
    # for i in range(1,51):
    #     knnStor.append(i)
    # for k in knnStor:
    #     knn = Knn(k)
    #     knn.train(xTrain, yTrain['label'])
    #     # predict the training dataset
    #     yHatTrain = knn.predict(xTrain)
    #     trainAcc = accuracy(yHatTrain, yTrain['label'])
    #     # predict the test dataset
    #     yHatTest = knn.predict(xTest)
    #     testAcc = accuracy(yHatTest, yTest['label'])    

    #     kVal.append(k)
    #     teAcc.append(testAcc)
    #     trAcc.append(trainAcc)

    #     print("K Value:", k)
    #     print("Training Acc:", trainAcc)
    #     print("Test Acc:", testAcc)
    #     print()


    # print(kVal)
    # plt.plot(kVal,teAcc,'r--', label = 'Test Accuracy')
    # plt.plot(kVal,trAcc, 'b--', label = 'Train Accuracy' )
    # plt.xlabel('K Values')
    # plt.ylabel('Accuracy')


    # plt.legend()

    # plt.show()

if __name__ == "__main__":
    main()
