import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # TODO do more than this

    # Extract a infp then use function to return a numeric for the day of the week
    # Then converted the numeric to binary value to 0 or 1
    df['weekType'] = pd.to_datetime(df.date)

    df.weekType = df.weekType.dt.dayofweek

    df.weekType //= 5

    df = df.drop(columns=['date'])

    return df

def graphMatrix(matrix):
    sns.set()

    # Graph the correlation matrix into heatmap
    ax = sns.heatmap(matrix)

    plt.show()

def pearsonMatrix(dfTemp, df):
    featCol = []

    # Appended the features to an array then calculate pearson matrix using numpy
    for i in range(len(dfTemp[0])):
        featCol.append(dfTemp[:,i])

    pearsonMatrix = np.corrcoef(featCol)

    pearsonMatrix = pd.DataFrame(pearsonMatrix)

    pearsonMatrix.index, pearsonMatrix.columns = df.columns, df.columns

    return pearsonMatrix

def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # TODO

    
    dfTemp = np.asarray(df)
    matrix = pearsonMatrix(dfTemp, df)

    # graphMatrix(matrix)
    print(matrix)

    matrix = np.asarray(matrix)

    deleteColBool = np.full( (1,len(dfTemp[0])), True )

    deleteColBool = deleteColBool[0]

    # Goes through the correlation matrix and marks the ones with high correlation
    for i in range(len(dfTemp[0])):
        for t in range(i + 1, len(dfTemp[0])):
            if( abs(matrix[i,t]) >= .75):
                deleteColBool[t] = False

    
    # Drops the columns marked 
    deleteColIndex = []
    for i in range(len(deleteColBool)):
        if (deleteColBool[i] == False):
            deleteColIndex.append(i)
    
    df = df.drop(df.columns[deleteColIndex], axis = 1)
    

    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    colTrain = trainDF.columns
    colTest = testDF.columns

    # Uses sklearn to preprocess the data
    min_max_scaler = preprocessing.MinMaxScaler()

    trainDF, testDF = min_max_scaler.fit_transform(trainDF), min_max_scaler.fit_transform(testDF)

    trainDF = pd.DataFrame(trainDF)
    trainDF.columns = colTrain

    testDF = pd.DataFrame(testDF)
    testDF.columns = colTest


    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
