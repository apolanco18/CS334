import argparse
import numpy as np
import pandas as pd
from math import log
from sklearn.metrics import accuracy_score
import time
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class NodeHolder(object):
    data = None
    leftNodeHolder = None
    rightNodeHolder = None
    depthLevel = 0
    optValue = 0
    colIndex = 0

    def _init_(self,data):
        self.data = data

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def count(self, data):
        # finds the most common value in the binary array
        (values,counts) = np.unique(data[:,len(data[0])-1],  return_counts=True)
 
        mostCom = values[np.argmax(counts)]
     
        count = 0
        # Finds the count of the return most common value
        for yVal in data:
            if mostCom == yVal[len(data[0]) - 1]:
                count += 1

        leastCom = 0
        if(mostCom == 0):  
            leastCom = 1
        
        # Finds the least common value and count
        leastComCount = len(data) - count

        return mostCom, count, leastCom, leastComCount
    
    def entropy(self, data, parentNodeLen, featIndex):
        if data is None:
            return 0
        
        mostCom, comCount, leastCom, leastComCount = self.count(data)

        # Calculate the entrooy value
        if(leastComCount == 0 or comCount == 0):
            return 1
        else:
            return - (float(len(data))/parentNodeLen) * ( (float(comCount)/len(data)) *  log((float(comCount)/len(data)),2)  + (float(leastComCount)/len(data)) * log((float(leastComCount)/len(data)),2 ) )
        
    
    def gini(self, data, parentNodeLen, featIndex):
        if data is None:
            return 0
        mostCom, comCount, leastCom, leastComCount = self.count(data)


        # Calculate the gini value
        if(leastComCount == 0 or comCount == 0):
            return 1
        else:
            return (float(len(data))/parentNodeLen) * ( (float(comCount)/len(data)) *  (float(leastComCount)/len(data))  + (float(leastComCount)/len(data)) * (float(comCount)/len(data)) )

    
    def split(self, xFeat, featIndex , val ):
        # Splits the nodes based on thresold value
        leftNodeData, rightNodeData = list(),list()
        for row in xFeat:
            if row[featIndex] < val:
                leftNodeData.append(row)
            else:
                rightNodeData.append(row)
        
        leftNodeData = np.asarray(leftNodeData)
        if(leftNodeData.size < self.minLeafSample):
            leftNodeData = None

        rightNodeData = np.asarray(rightNodeData)
        if(rightNodeData.size < self.minLeafSample):
            rightNodeData = None
        
        
        return leftNodeData,rightNodeData

    def method(self, data, parentNodeLen, index):
        # Checks the user inputed method then directs it program to compute specific method
        if (self.criterion == 'entropy'):
            return self.entropy(data,parentNodeLen,index)
        elif(self.criterion == 'gini'):
            return self.gini(data,parentNodeLen,index)


    def build_Tree(self, nodeHolder):
        xFeat = nodeHolder.data

        if(nodeHolder.depthLevel < self.maxDepth):
            finalCritVal = 99999999
            finalCritIndex = 0
            finalColIndex = 0
            # goes through every possible value and tests each value as optimal thresold
            for i in range(len(xFeat[0]) - 1):
                critVal = 99999999
                critValIndex = 0
                prevVal = 0
                rowCount = 0
                for row in xFeat:
                    if(rowCount + 1 == len(xFeat) or xFeat[rowCount + 1,len(xFeat[0])-1] != row[i]):
                        left, right = self.split(xFeat,i,row[i])
                            
                        tempCritVal = self.method(left,len(xFeat),i) + self.method(right,len(xFeat),i)

                        if tempCritVal == 0:
                            tempCritVal = 1

                        if critVal > tempCritVal:
                            critVal = tempCritVal
                            critValIndex = row[i]
                    
                    rowCount += 1

                if critVal < finalCritVal:
                    finalCritVal = critVal
                    finalCritIndex = critValIndex
                    finalColIndex = i


            # Stores the optimal threshold info then splits the node and recursively calls itself
            nodeHolder.optValue = finalCritIndex
            nodeHolder.colIndex = finalColIndex

            leftNodeData, rightNodeData = self.split(xFeat,finalColIndex,finalCritIndex)

            tempL , tempR = NodeHolder(),NodeHolder()

            tempL.data = leftNodeData
            tempR.data = rightNodeData

            level = nodeHolder.depthLevel + 1
            tempL.depthLevel, tempR.depthLevel = level, level


            if(tempL.data is not None):
                nodeHolder.leftNodeHolder = tempL
                if tempL.depthLevel <= self.maxDepth: 
                    self.build_Tree(tempL)
            else:
                nodeHolder.leftNodeHolder = None
            if(tempR.data is not None):
                nodeHolder.rightNodeHolder = tempR
                if tempR.depthLevel <= self.maxDepth:
                    self.build_Tree(tempR)
            else:
                nodeHolder.rightNodeHolder = None
        else:
            nodeHolder.leftNodeHolder = None
            nodeHolder.rightNodeHolder = None
  

    def train(self, xFeat, y):
        """
        Train the decision tree model.

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
        xFeat['yVal'] = y
        xFeat =  xFeat.to_numpy()

        global root 
        root = NodeHolder()
        root.data = xFeat
        self.build_Tree(root)

        return self 


    def predict_Val(self,row, nodeHolder):

        # Recursively goes through the tree to find specific node value
    
        if row[nodeHolder.colIndex] < nodeHolder.optValue:

            if nodeHolder.leftNodeHolder is not None:
                return self.predict_Val(row,nodeHolder.leftNodeHolder)
            else:
                mostCom, comCount, leastCom, leastComCount = self.count(nodeHolder.data)
                return mostCom
        else:
            if nodeHolder.rightNodeHolder is not None:
                return self.predict_Val(row,nodeHolder.rightNodeHolder)
            else:
                mostCom, comCount, leastCom, leastComCount = self.count(nodeHolder.data)
                return mostCom


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
        xFeat = xFeat.to_numpy()

        global root
        

        for row in xFeat:
            yHat.append((self.predict_Val(row,root)))
        
        
        return yHat

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc

def plot_accuracy(xTrain,yTrain,xTest,yTest):
    minLeaf = []
    maxDepth = []
    accTrain = []
    accTest = []

    # plots the accuracy in 3-d plot

    for i in range(16):
        for t in range(10,40,10):
            dt  = DecisionTree('gini',i,t)
            trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
            minLeaf.append(t)
            maxDepth.append(i)
            accTrain.append(trainAcc)
            accTest.append(testAcc)

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(minLeaf,maxDepth, accTrain, c = '#e11e72',label = 'Train Accuracy')
    ax.scatter(minLeaf,maxDepth, accTest, c = '#26d97e', label = 'Test Accuracy')
    
    ax.set_xlabel('Min Leaf Sample')
    ax.set_ylabel('Max Depth Level')
    ax.set_zlabel('Accuracy')
    ax.legend()

    plt.show()

root = None


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)

    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


    # plot_accuracy(xTrain,yTrain,xTest,yTest)



if __name__ == "__main__":
    main()
