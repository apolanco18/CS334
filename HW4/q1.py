import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def model_assessment(model,x_train,x_test,y_train,y_test):
    """
    Given the entire data, decide how
    you want to assess your different models
    to compare perceptron, logistic regression,
    and naive bayes, the different parameters, 
    and the different datasets.
    """

    model.fit(x_train,y_train)

    yPredict = model.predict(x_test)
    acc = accuracy_score(yPredict,y_test)
    numMistake = (1 - acc) * len(y_test)
    return numMistake


def build_vocab_map(data):

    vectorizer = CountVectorizer(binary = True)
    X = vectorizer.fit_transform(data)
    colNames = vectorizer.get_feature_names()
    
    count = X.sum(axis = 0)
    validArr = []
    for index, val in enumerate(np.nditer(count)):
        if(val > 30):
            validArr.append(index)

    
    colNames = [colNames[i] for i in validArr]
   
    return colNames

def findRowCount(filename):
    f = open(filename,"r")
    count = 0

    for line in f:
        count += 1
    return count

def construct_binary(data, words):
    """
    Construct the email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise


    """
    vectorizer = CountVectorizer(binary = True)
    vectorizer.fit_transform(words)
    binaryMap = vectorizer.transform(data)

    binaryMap = np.asarray(binaryMap.toarray())

    binaryMap = pd.DataFrame(binaryMap,columns = words)

    return binaryMap


def construct_count(data,words):
    """
    Construct the email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    vectorizer = CountVectorizer()
    vectorizer.fit_transform(words)
    countMap = vectorizer.transform(data)

    countMap = np.asarray(countMap.toarray())

    countMap = pd.DataFrame(countMap,columns = words)
    return countMap


def construct_tfidf(data, words):
    """
    Construct the email datasets based on
    the TF-IDF representation of the email.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(words)
    tfMap = vectorizer.transform(data)

    tfMap = np.asarray(tfMap.toarray())

    tfMap = pd.DataFrame(tfMap,columns = words)
    

    return tfMap

def splitData(map,label):
    x_train, x_test, y_train, y_test = train_test_split(map,label,test_size = .3)

    return x_train,x_test,y_train,y_test

def ExportData(x_train,x_test,y_train,y_test,name):

    f1= open(name + "_xtrain.csv","w+")
    x_train.to_csv(f1,index = False)
    f2 = open(name + "_xtest.csv","w+")
    x_test.to_csv(f2,index = False)

    f3 = open(name + "_ytrain.csv","w+")
    y_train.to_csv(f3,index = False)
    f4 = open(name + "_ytest.csv","w+")
    y_test.to_csv(f4,index = False)


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    # model_assessment(args.data)
    y = []

    f = open(args.data,"r")
    data = []

    for line in f:
        y.append(line[0])
        data.append(line)

    y = pd.DataFrame(y)

    
        
    x_train,x_test,y_train,y_test = splitData(data,y)

    words = build_vocab_map(x_train)

    binaryMapTrain = construct_binary(x_train,words)
    binaryMapTest = construct_binary(x_test,words)
    countMapTrain = construct_count(x_train,words)
    countMapTest = construct_count(x_test,words)
    tfMapTrain = construct_tfidf(x_train,words)
    tfMapTest = construct_tfidf(x_test,words)


    ExportData(binaryMapTrain,binaryMapTest,y_train,y_test,"binary")
    ExportData(countMapTrain,countMapTest,y_train,y_test,"count")
    ExportData(tfMapTrain,tfMapTest,y_train,y_test,"tf")


    # This the code I used to evaluate Q3 I used a model assesment to evaluate Naive Bayes and Linear Regression
    # on the three different datasets

    # model = MultinomialNB()    
    # print("----Naive Bayes------")

    # numMistake = model_assessment(model,binaryMapTrain,binaryMapTest,y_train,y_test)
    # print("Binary Map")
    # print("Number of Mistakes: {}".format(numMistake))

    # numMistake = model_assessment(model,countMapTrain,countMapTest,y_train,y_test)
    # print("Count Map")
    # print("Number of Mistakes: {}".format(numMistake))

    # numMistake = model_assessment(model,tfMapTrain,tfMapTest,y_train,y_test)
    # print("TF Map")
    # print("Number of Mistakes: {}".format(numMistake))


    # model = LogisticRegression(solver="lbfgs", max_iter = 500)
    # print("----Linear Regression------")

    # numMistake = model_assessment(model,binaryMapTrain,binaryMapTest,y_train,y_test)
    # print("Binary Map")
    # print("Number of Mistakes: {}".format(numMistake))

    # numMistake = model_assessment(model,countMapTrain,countMapTest,y_train,y_test)
    # print("Count Map")
    # print("Number of Mistakes: {}".format(numMistake))

    # numMistake = model_assessment(model,tfMapTrain,tfMapTest,y_train,y_test)
    # print("TF Map")
    # print("Number of Mistakes: {}".format(numMistake))



if __name__ == "__main__":
    main()
