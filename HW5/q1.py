import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing


# Normalize the data by scaling the data to represent Gaussian data with zero mean and unit variance
def normalize_data(xTrain,xTest):
    xTrainScaled = preprocessing.scale(xTrain)
    xTestScaled = preprocessing.scale(xTest)

    return xTrainScaled,xTestScaled


# def unreg_logistic_regression(xTrain,yTrain,xTest,yTest):


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

    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    xTrain, xTest = normalize_data(xTrain,xTest)

    print(xTrain)
    print(xTest)

    





if __name__ == "__main__":
    main()
