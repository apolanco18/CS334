import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from standardLR import StandardLR
from sgdLR import SgdLR
from lr import file_to_numpy
import random

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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed) 

    batchSize = [1,10,30,130,215,640,16770]
    lr = [.01,1,10,100,100,1000,100000]

    # Goes through a list of batch sizes and ideal learning rates and graphs the required information
    for i in range(len(batchSize) ):
        model = SgdLR(lr[i],batchSize[i], args.epoch)
        trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)

        time = []
        trainMse = []
        testMse = []
        

        for key in trainStats:
            temp = trainStats[key]
            time.append(temp['time'])
            trainMse.append(temp['train-mse'])
            testMse.append(temp['test-mse'])

    
        plt.subplot(2,1,1)

        color = '#' + "%06x" % random.randint(0, 0xFFFFFF)
        plt.scatter(time,trainMse,c = color, label = 'BS: %d ' % (batchSize[i]) )

        plt.subplot(2,1,2)

        color = '#' + "%06x" % random.randint(0, 0xFFFFFF)
        plt.scatter(time,testMse,c = color, label = 'BS: %d ' % (batchSize[i]) )  

    
    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)

    plt.subplot(2,1,1)
    color = '#' + "%06x" % random.randint(0, 0xFFFFFF)
    plt.plot( (trainStats[0])['time'],  (trainStats[0])['train-mse'], c = color, label = 'Closed Form', marker = 'o', markerSIze = 12)

    plt.subplot(2,1,2)
    color = '#' + "%06x" % random.randint(0, 0xFFFFFF)
    plt.plot( (trainStats[0])['time'],  (trainStats[0])['test-mse'], c = color, label = 'Closed Form',  marker = 'o', markerSIze = 12)


    
    plt.subplot(2,1,1)

    plt.xscale('log')
    plt.xlabel('Total Time')
    plt.ylabel('MSE')
    plt.title('Training-MSE')
    plt.legend()

    plt.subplot(2,1,2)

    plt.xscale('log')
    plt.xlabel('Total Time')
    plt.ylabel('MSE')
    plt.title('Test-MSE')
    plt.legend()

    plt.show()
  



if __name__ == "__main__":
    main()