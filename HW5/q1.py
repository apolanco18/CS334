import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn import metrics
import matplotlib.pyplot as plt


# Normalize the data by scaling the data to represent Gaussian data with zero mean and unit variance
def normalize_data(xTrain,xTest):
    xTrainScaled = StandardScaler().fit_transform(xTrain)
    xTestScaled = StandardScaler().fit_transform(xTest)

    return xTrainScaled,xTestScaled

# Performs a Unregularizes Logistic Regression on a dataset and returns the predicted probablities
def unreg_logistic_regression(xTrain,yTrain,xTest,yTest):
    model = LogisticRegression(penalty = 'none',solver = 'lbfgs')
    
    model.fit(xTrain,yTrain)

    prob = model.predict_proba(xTest)

    return prob[:,1]

def unregLogNorm(xTrain,yTrain,xTest,yTest):
    model = LogisticRegression(penalty = 'none',solver = 'lbfgs')
    model.fit(xTrain,yTrain)

    yPredict = model.predict_proba(xTest)

    yPredict = yPredict[:,1]

    return metrics.roc_curve(yTest,yPredict)

def unregLogPCA(xTrain,yTrain,xTest,yTest):
    model = PCA(.95)

    model.fit(xTrain)

    xTrainTrans = model.transform(xTrain)
    xTestTrans = model.transform(xTest)

    modelLog = LogisticRegression(penalty = 'none',solver = 'lbfgs')

    modelLog.fit(xTrainTrans,yTrain)

    yPredict = modelLog.predict_proba(xTestTrans)

    yPredict = yPredict[:,1]

    return metrics.roc_curve(yTest,yPredict)

def unregLogNMF(xTrain,yTrain,xTest,yTest):
    model = NMF(n_components = 9)

    model.fit(xTrain)

    xTrainTrans = model.transform(xTrain)
    xTestTrans = model.transform(xTest)

    modelLog = LogisticRegression(penalty = 'none',solver = 'lbfgs')

    modelLog.fit(xTrainTrans,yTrain)

    yPredict = modelLog.predict_proba(xTestTrans)

    yPredict = yPredict[:,1]

    return metrics.roc_curve(yTest,yPredict)



# Read an input file and convert it to numpy array
def file_to_numpy(filename):
    temp = pd.read_csv(filename)
    return temp.to_numpy()

def main():
    
    parser = argparse.ArgumentParser()

    # Read datasets as arguments from the command line
    parser.add_argument("xTrain",help = "filename for features of the training data")
    parser.add_argument("yTrain",help = "filename for labels associated with the training data")
    parser.add_argument("xTest",help = "filename for features of the test data")
    parser.add_argument("yTest",help = "filenamse for labels associated with the test data")
    args = parser.parse_args()

    # Parse the data and convert into numpy array
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)


    # Normalize the feature matrix in both train and test dataset
    xTrainNorm, xTestNorm = normalize_data(xTrain,xTest)

    unregLogProb = unreg_logistic_regression(xTrain,yTrain,xTest,yTest)

    # print(unregLogProb)

    # This code was used to find the variance in which  it captures about 95% of the variance in the original data 
    model = PCA(.95)

    W = model.fit_transform(xTrainNorm)

    print(W)

    # print(model.explained_variance_ratio_.cumsum())

    # This prints all the principle components, and to find the best features the first 3 rows would be taken and then average. The features with the most positive weights are the answer
    H = model.components_
    print(H)


    # I calculated the reconstruction error for a range of components and the I realized that
    # the lower number of components is better for the model. So look through the range ann saw that 9
    # components was a good value to use
    # for i in range(1,21):
    #     model = NMF(n_components = i)

    #     test = model.fit(xTrain)

    #     print(test.reconstruction_err_)

    # This code graphs the roc curve of all the data
    rocNormfpr, rocNormtpr, thresholds = unregLogNorm(xTrainNorm,yTrain,xTestNorm,yTest)
    # print(rocNormfpr)
    # print(rocNormtpr)

    rocPCAfpr, rocPCAtpr, thresholds1 = unregLogPCA(xTrainNorm,yTrain,xTestNorm,yTest)
    # print(rocPCAfpr)
    # print(rocPCAtpr)

    rocMNFfpr, rocMNFtpr, thresholds2 = unregLogPCA(xTrain,yTrain,xTest,yTest)
    # print(rocMNFfpr)
    # print(rocMNFtpr)

    fig = plt.figure()
    ax = fig.add_subplot(111,)
    ax.scatter(rocNormfpr,rocNormtpr, c = '#e11e72',label = 'Norm Data')
    ax.scatter(rocPCAfpr,rocPCAtpr, c = '#26d97e', label = 'PCA Data')
    ax.scatter(rocMNFfpr,rocMNFtpr, c = '#4279bd', label = 'MNF Data')

    
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    ax.legend()

    plt.show()



    





if __name__ == "__main__":
    main()
