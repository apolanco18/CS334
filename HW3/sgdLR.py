import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import random 

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch
    
    def sgd(self, x , y, type, trainStats):

        beta = np.random.uniform(size = (len(x[0])))

        timeE = 0
        # Iterates through the all the epochs
        for epoch in range(self.mEpoch):
            tempTimeElaspe = time.time()
            xRand, yRand = LinearRegression.shuffle(self,x,y)

            # Splits the data into equal batches
            for b in range(0, len(x) , self.bs):

                xSub = xRand[b: b + self.bs]
                ySub = yRand[b: b + self.bs]

                gradAvg = 0
                
                # For the batch follows the equation Y - XB then I average all the batches into one sample
                part = ( ySub - xSub * beta)
                part = part.mean(0)

                xTemp = xSub
                xTemp = xTemp.mean(0)

                gradAvg = xTemp * part   
                
                gradAvg /= float(len(xSub))


                # Update beta based on the calucaltions done previously
                beta = beta + ((self.lr/len(x)) * (gradAvg) )
   

            LinearRegression.beta = beta
            
            tempTimeElaspe = time.time() - tempTimeElaspe
            temp = trainStats[epoch * self.bs]
            timeE += tempTimeElaspe
            temp['time'] += timeE 

            if(type == 'train'):
                temp['train-mse'] = LinearRegression.mse(self,x,y)
            elif(type == 'test'):
                temp['test-mse'] = LinearRegression.mse(self,x,y)

    
    def find_opt_learn(self,xTrain,yTrain):

        index = np.random.choice(len(xTrain), round(.4 *(len(xTrain))) )
        xRSub = xTrain[index]
        yRSub = yTrain[index]
        trainStats = {}

        lrEpoch = []
        lrMse = [] 

        
        ax = plt

        # Goes through set of learning rates and plots them
        for lr in [.1, .01, .001, .0001]:
            self.lr = lr
            lrEpoch = []
            lrMse = []
            for epoch in range(self.mEpoch):
                trainStats.update( {epoch * self.bs: {'time':0,'train-mse':0,'test-mse':0} } )

            self.sgd(xTrain,yTrain, 'train',trainStats)

            for key in trainStats:
                temp = trainStats[key]
                lrEpoch.append(key)
                lrMse.append(temp['train-mse'])
            

            color = '#' + "%06x" % random.randint(0, 0xFFFFFF)

            ax.scatter(lrEpoch,lrMse, c = color, label = '%f Learning Rate' % lr)



        ax.xlabel('Epoch')
        ax.ylabel('Train-MSE')
        ax.legend()

        plt.yscale('log')
        plt.show()

    def graph_opt_learn(self,xTrain,yTrain,xTest,yTest):

        Epoch = []
        trainMse = []
        testMse = []
        trainStats = {}

        for epoch in range(self.mEpoch):
            trainStats.update( {epoch * self.bs: {'time':0,'train-mse':0,'test-mse':0} } )
        
        self.sgd(xTrain,yTrain, 'train',trainStats)
        self.sgd(xTrain,yTrain, 'test',trainStats)

        ax = plt

        # Runs the sgd linear regression and graphs all the mse values
        for key in trainStats:
            temp = trainStats[key]
            Epoch.append(key)
            trainMse.append(temp['train-mse'])
            testMse.append(temp['test-mse'])

        
        color = '#' + "%06x" % random.randint(0, 0xFFFFFF)

        ax.scatter(Epoch,trainMse, c = color, label = 'Train-Mse' )

        color = '#' + "%06x" % random.randint(0, 0xFFFFFF)

        ax.scatter(Epoch,testMse, c = color, label = 'Test-Mse' )
        
        
        ax.xlabel('Epoch')
        ax.ylabel('MSE')
        ax.legend()

        plt.yscale('log')
        plt.show()

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD
        
        for epoch in range(self.mEpoch):
            trainStats.update( {epoch * self.bs: {'time':0,'train-mse':0,'test-mse':0} } )

        
        # self.find_opt_learn(xTrain,yTrain)
        # self.graph_opt_learn(xTrain,yTrain,xTest,yTest)

        self.sgd(xTrain,yTrain, 'train',trainStats)
        self.sgd(xTest,yTest, 'test',trainStats)


        return trainStats


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
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

