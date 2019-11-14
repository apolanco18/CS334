import q1
import pandas as pd
import numpy as np
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def naiveBayes(data, labels):
	model = MultinomialNB()
	score, matrix = q1.model_assessment(model, data, labels)
	print("f1 score of Naive Bayes: {}".format(score))
	print("Confusion Matrix:")
	print(matrix)
	print("")


def logisticRegression(data, labels):
	model = LogisticRegression(solver="lbfgs", max_iter = 500)
	score, matrix = q1.model_assessment(model, data, labels)
	print("f1 score of Logistic Regression: {}".format(score))
	print("Confusion Matrix:")
	print(matrix)
	print("")

def parseData(name): #loads the labels from the spamassassin.data file
	data = pd.read_csv(name)
	labels=[]
	data = data.values.tolist()
	for i in range(len(data)):
		labels.append(int(data[i][0][0]))
	return labels

def main():
	labels = parseData("spamAssassin.data")
	binaryData = pd.read_csv("binaryVector.csv", header=None)
	countData = pd.read_csv("countVector.csv", header=None)
	tfIdfData = pd.read_csv("tfidfVector.csv", header=None)
	
	print("--------------Binary Data:--------------")
	naiveBayes(binaryData, labels)
	logisticRegression(binaryData, labels)

	print("--------------Counted Data:--------------")
	naiveBayes(countData, labels)
	logisticRegression(countData, labels)

	print("--------------tfidf Data:--------------")
	naiveBayes(tfIdfData, labels)
	logisticRegression(tfIdfData, labels)



if __name__ == "__main__":
	main()
