
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np


iris = datasets.load_iris()


features = iris.data[:,[0,1,2,3]]


## Partion the data into segments needed for graphing 
irisS = [features[0:50,0],features[0:50,1],features[0:50,2],features[0:50,3]]
irisVer = [features[50:100,0],features[50:100,1],features[50:100,2],features[50:100,3]]
irisVirg = [features[100:150,0],features[100:150,1],features[100:150,2],features[100:150,3]]

## Function to set color for individual boxplot
def setBoxPlotColor(plotVar, color):
    plt.setp(plotVar['boxes'], color = color)
    plt.setp(plotVar['whiskers'], color= color)
    plt.setp(plotVar['caps'], color= color)
    plt.setp(plotVar['medians'], color= color)  

plt.figure()

## Plots each of the boxplots
pl1 = plt.boxplot(irisS, sym='', widths=0.4)
pl2 = plt.boxplot(irisVer, sym='', widths=0.4)
pl3 = plt.boxplot(irisVirg, sym='', widths=0.4)

setBoxPlotColor(pl1,'#1ecbe1')
setBoxPlotColor(pl2,'#6c33cc')
setBoxPlotColor(pl3,'#e62519')

plt.plot([], c = '#1ecbe1', label = 'Iris Setosa')
plt.plot([], c = '#6c33cc', label = 'Iris Versicolour')
plt.plot([], c = '#e62519', label = 'Iris Virginica')
plt.legend()

plt.tight_layout()
plt.xticks(np.arange(5),('','sepalLength','sepalWidth','petalLength','petalWidth'))


fig = plt.subplots()
ax = plt.subplot()


## A Loop to go through each species type to graph on a scatter plot

for pos in [1,2,3]:
    if pos == 1 :
        x = features[0:50,0]
        y = features[0:50,1]
        ax.scatter(x,y,c = '#1ecbe1',label = 'Iris Setosa',alpha = .3)
    elif pos == 2 :
        x = features[50:100,0]
        y = features[50:100,1]
        ax.scatter(x,y,c = '#6c33cc',label = 'Iris Versicolour',alpha = .3)
    elif pos == 3:
        x = features[100:150,0]
        y = features[100:150,1]
        ax.scatter(x,y,c = '#e62519',label = 'Iris Virginica',alpha = .3)

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.legend()



fig1 = plt.subplots() 
ax1 = plt.subplot()

for pos in [1,2,3]:
    if pos == 1 :
        x = features[0:50,1]
        y = features[0:50,2]
        ax1.scatter(x,y,c = '#1ecbe1',label = 'Iris Setosa',alpha = .3)
    elif pos == 2 :
        x = features[50:100,1]
        y = features[50:100,2]
        ax1.scatter(x,y,c = '#6c33cc',label = 'Iris Versicolour',alpha = .3)
    elif pos == 3:
        x = features[100:150,1]
        y = features[100:150,2]
        ax1.scatter(x,y,c = '#e62519',label = 'Iris Virginica',alpha = .3)

ax1.set_xlabel('Petal Length')
ax1.set_ylabel('Petal Width')
ax1.legend()


plt.show()