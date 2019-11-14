import argparse
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import time
import random
import matplotlib.pyplot as plt


def find_opt_hyper(xFeat, y):

    xFeat = np.asarray(xFeat)
    y = np.asarray(y)

    kf = KFold(n_splits = 10)
    kf.get_n_splits(xFeat)

    finalTestAucKNN = 0
    finalKVal = 0
    finalTestAucTree = 0
    finalMaxDepth = 0
    finalMinSample = 0
    finalType = None


    for train_index, test_index in kf.split(xFeat):
        xTrain, xTest = xFeat[train_index], xFeat[test_index]
        yTrain, yTest = y[train_index], y[test_index]

        
        for i in range(1,26):
            kn = KNeighborsClassifier(n_neighbors = i)
            kn.fit(xTrain,yTrain)

            predictTest = kn.predict_proba(xTest)

            fpr, tpr, thresholds = metrics.roc_curve(yTest,predictTest[:, 1])
            testAuc = metrics.auc(fpr, tpr)

    

            if testAuc > finalTestAucKNN:
                finalTestAucKNN = testAuc
                finalKVal = i
            for t in range(10,65,10):
                for f in ['gini','entropy']:
                    dt = DecisionTreeClassifier(f,max_depth = i, min_samples_leaf = t,)

                    dt.fit(xTrain,yTrain)

                    predictTest = dt.predict_proba(xTest)

                    fpr, tpr, thresholds = metrics.roc_curve(yTest,predictTest[:, 1])
                    testAuc = metrics.auc(fpr, tpr)

                    if testAuc > finalTestAucTree:
                        finalTestAucTree = testAuc
                        finalMaxDepth = i
                        finalMinSample = t
                        finalType = f




        print("K-Value",finalKVal)
        print("Test AUC Knn", finalTestAucKNN)
        print("-----------")

        print("Max Depth",finalMaxDepth)
        print("Min Sample", finalMinSample)
        print("Method",finalType)
        print("Test AUC tree", finalTestAucTree)
        print("-----------")

        return finalKVal, finalMaxDepth, finalMinSample, finalType


def remove(n,xTrain, yTrain):

    # Removes random info from data set
    for i in range( int (len(xTrain) * n) ):
        val = random.randrange(0,len(xTrain) - 1)
        xTrain = np.delete(xTrain, val,  axis = 0)
        yTrain = np.delete(yTrain, val,  axis = 0)
    
    return xTrain, yTrain


#  Fits the data and generates a prediction
def perform_test(model, xTrain, yTrain, xTest, yTest):
    model = model.fit(xTrain,yTrain)

    predictTestAuc = model.predict_proba(xTest)

    fpr1, tpr1, thresholds = metrics.roc_curve(yTest,predictTestAuc[:,1])

    predictTestAcc = model.predict(xTest)


    return metrics.auc(fpr1,tpr1), metrics.accuracy_score(yTest,predictTestAcc)


def optimal(k, maxD, minSample, type, xTrain,yTrain,xTest,yTest):


    # Loops through data to run test
    xTrain = np.asarray(xTrain)
    yTrain = np.asarray(yTrain)
    xTest = np.asarray(xTest)
    yTest = np.asarray(yTest)

    model = KNeighborsClassifier(n_neighbors = k)
    model1 = DecisionTreeClassifier(type,max_depth = maxD, min_samples_leaf = minSample)

    aucKnn, accKnn = perform_test(model, xTrain, yTrain, xTest, yTest)
    aucTree, accTree = perform_test(model1, xTrain, yTrain, xTest, yTest)


    print("Data Removed: 0%")
    print("AUC %s" % aucKnn)
    print("Accuracy %s" % accKnn)

    
    aucK = []
    accK = []
    aucT = []
    accT = []

    aucK.append(aucKnn)
    accK.append(accKnn)

    aucT.append(aucTree)
    accT.append(accTree)

    


    perform_test(model,xTrain,yTrain,xTest,yTest)


    for i in [.01,.05,.10]:
        xTrain1, yTrain1, = remove(i,xTrain, yTrain)
        xTest1, yTest1 =  xTest, yTest

        aucKnn, accKnn = perform_test(model,xTrain1,yTrain1, xTest1, yTest1)

        aucTree, accTree = perform_test(model1,xTrain1,yTrain1, xTest1, yTest1)

        
        aucK.append(aucKnn)
        accK.append(accKnn)

        aucT.append(aucTree)
        accT.append(accTree)
        
    
    
    d = {'Data Removed':['0%','.01%','.05%','.1%'],'Auc Knn':aucK, 'Acc Knn': accK,'Auc Tree':aucT, 'Acc Tree':accT}
    df = pd.DataFrame(data = d)


    print('')
    print('')
    print('')

    print('Optimal K Value %d' % k)
    print('Optimal Max Depth %d' % maxD)
    print('Optimal Min Leaf Sample %d' % minSample)
    print('Optimal Type %s' % type)
    
    print('------------')
    print(df)
    print('------------')

   

    






def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()

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



    optK , optMaxDepth, optMinSample, optFinalType = find_opt_hyper(xTrain,yTrain)
    optimal(optK, optMaxDepth, optMinSample, optFinalType, xTrain,yTrain,xTest,yTest)
    




if __name__ == "__main__":
    main()
