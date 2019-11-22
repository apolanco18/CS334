import argparse
import numpy as np
import pandas as pd
from random import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from statistics import mean 
from tabulate import tabulate




class RandomForest(object):
    nest = 0          
    maxFeat = 0        
    maxDepth = 0       
    minLeafSample = 0  
    criterion = None   
    trees = []

    def __init__(self, nest, maxFeat=None, criterion=None, maxDepth=None, minLeafSample=None):

        self.nest = nest
        self.criterion = criterion
        self.maxFeat = maxFeat
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def bagData(self,xFeat,yFeat):
        xFeatBag, yFeatBag = [],[]
        indexes = []

        for i in range(len(xFeat)):
            randomIndex = randint(0,len(xFeat) - 1)

            xFeatBag.append(xFeat[randomIndex])
            yFeatBag.append(yFeat[randomIndex])
            indexes.append(randomIndex)

        return xFeatBag,yFeatBag,indexes

    def calAcc(self,yFeat,yHat):
        correct = 0
        for i in range(len(yFeat)):
            if yFeat[i] == yHat[i]:
                correct += 1
        return correct / len(yFeat)

    def calculateOobError(self,trees,xFeat,yFeat):
        maxNum = 0
        correctNum = 0

        for i in range(len(xFeat)):
            for info in trees:
                if i not in info[0]:
                    if info[1].predict([xFeat[i]]) == yFeat[i]:
                        correctNum += 1
                    maxNum += 1
        
        result = (maxNum - correctNum)/maxNum

        return result

    
    def train(self, xFeat, y):
        

        numTrees = []
        stats = {}
        error = []
        errorNum = []

        for numSizes in range(self.nest):
            model = DecisionTreeClassifier(criterion = self.criterion, max_features = self.maxFeat, max_depth = self.maxDepth, min_samples_leaf = self.minLeafSample)

            xFeatBag,yFeatbag,indexes = self.bagData(xFeat,y)

            treeInfo = []
            treeInfo.append(indexes)
            treeInfo.append(model)

            numTrees.append(treeInfo)

            errorNum.append(numSizes)
            model.fit(xFeatBag,yFeatbag)
            error.append(self.calculateOobError(numTrees,xFeat,y))
        
        for i in range(len(errorNum)):
            stats[i] = error[i]
            
        self.trees = numTrees

        return stats

    def predict(self, xFeat):
        yPredict = []

        for row in xFeat:
            predict = []
            for info in self.trees:
                prediction = info[1].predict([row])
                predict.append(prediction[0])
            yPredict.append(1) if mean(predict) > .5 else yPredict.append(0)
        
        return yPredict


def print_table(xTrain,yTrain,xTest,yTest):

    results = []

    for maxFeat in range(1,11):
        for size in range(1,20):
            model = RandomForest(size, maxFeat, 'gini', None, 1)
            trainStats = model.train(xTrain, yTrain)

            trainAcc = accuracy_score(yTrain, model.predict(xTrain))
            testAcc = accuracy_score(yTest,model.predict(xTest))

            results.append([maxFeat,size,trainAcc,testAcc])

    print(tabulate(results,headers = ["maxFear","size","trainAcc","testAcc"]))

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
    # parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()

    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    size = 2
    maxFeat = 10
    criterion = 'gini'
    maxDepth = None
    minLeafSample = 1  

    # print_table(xTrain,yTrain,xTest,yTest)

    
    model = RandomForest(size, maxFeat, criterion, maxDepth, minLeafSample) 
    trainStats = model.train(xTrain, yTrain)
    yHat = model.predict(xTest)
    print("Accuracy: ", accuracy_score(yTest, yHat))
    print("oobError: ", model.calculateOobError(model.trees, xTest,yTest)) 




if __name__ == "__main__":
    main()