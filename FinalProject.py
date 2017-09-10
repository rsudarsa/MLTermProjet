import csv
import pandas as pd

import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.linear_model import LogisticRegression

from pebl import data
from pebl.learner import greedy


def readCSVToDataFrame(filepath):
    f = open(filepath)
    readdata = csv.reader(f)
    data = []
    for row in readdata:
        data.append(row)
    data.pop(0)
# create a dataframe
    df = pd.DataFrame(data)
    return df
def estimaterRecognitionParameters(predictions, testY, model):
    correctlyClassified = 0
    singleDigitErrorRate = [0] * 10
    digitsFromTestData = [0] * 10
    for index in range(0, len(predictions)):
        if (testY[index] == predictions[index]):
            correctlyClassified += 1
            digitsFromTestData[testY[index]] = digitsFromTestData[testY[index]] + 1
        else:
            singleDigitErrorRate[testY[index]] = singleDigitErrorRate[testY[index]] + 1
    accuracy = float(correctlyClassified) * 100 / float(len(predictions))
    errorRate = (float(len(predictions)) - float(correctlyClassified)) * 100 / float(len(predictions))
    print ("Accuracy for ",model," : " ,accuracy)
    print ("ErrorRate " ,model," : " ,errorRate)
    for index in range(0,10):
        singleDigitError = float(singleDigitErrorRate[index]) * 100 / float(digitsFromTestData[index])
        print("Error Rate for the Digit ",index,": ", singleDigitError)



def trainNNBackPropagation(inputX, Y, solver, hiddenlayersizes,activation):
    # clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes = (45), activation = 'relu',verbose = False)
    clf = MLPClassifier(solver=solver, alpha=1e-5, hidden_layer_sizes = hiddenlayersizes, activation = 'relu',verbose = False)
    clf.fit(inputX,Y)
    return clf

def predictNNBackPropagation(testX,clf):
    predictions = clf.predict(testX)
    return predictions

def classifyUsingANN(inputX,Y,testX, testY, solver, hiddenlayersizes,activation):
    classifier = trainNNBackPropagation(inputX,Y,solver, hiddenlayersizes,activation)
    predictions = predictNNBackPropagation(testX,classifier)
    model = "ANN using "+str(hiddenlayersizes)+" hidden layers and "+ activation +" unit"
    estimaterRecognitionParameters(predictions,testY,model)

def createSVMClassifier(classifier):
    clf = SVC(kernel=classifier, verbose=False, decision_function_shape='ovo', cache_size=100)
    return clf


def classifyUsingSVM(inputX, Y, testX, testY,clf,model):
    clf.fit(inputX,Y)
    predictions = clf.predict(testX)
    estimaterRecognitionParameters(predictions, testY, model)

def classifyUsingKNN(inputX, Y, testX, testY, kValue):
    clf = KNeighborsClassifier(kValue)
    clf.fit(inputX,Y)
    predictions = clf.predict(testX)
    model = "KNN using "+str(kValue)+" clusters"
    estimaterRecognitionParameters(predictions, testY, model)

def classifyUsingNB(inputX, Y, testX, testY):
    clf = GaussianNB()
    clf.fit(inputX,Y)
    GaussianNB(priors=None)
    predictions = clf.predict(testX)
    model = "Naive Bayes"
    estimaterRecognitionParameters(predictions, testY,model)

def classifyUsingDecisionTree(inputX, Y, testX, testY):
    clf = tree.DecisionTreeClassifier()
    clf.fit(inputX,Y)
    predictions = clf.predict(testX)
    model  = "Decision Tree"
    estimaterRecognitionParameters(predictions, testY, model)

def classifyUsingLogisticRegression(inputX, Y, testX, testY,solver,multiclass):
    clf = LogisticRegression(solver=solver,multi_class=multiclass,max_iter=5000,random_state=23)
    clf.fit(inputX,Y)
    predictions = clf.predict(testX)
    model = "Logistic Regression"
    estimaterRecognitionParameters(predictions, testY , model)

#main program start
trainFileName = "data/optdigits.tra"
testFileName = "data/optdigits.tes"
trainFileNameForBayesian = "data/optdigits_tra_BN.txt"

inputDataFrame = readCSVToDataFrame(trainFileName)
testDataFrame = readCSVToDataFrame(testFileName)

classVariable = inputDataFrame.iloc[:,-1].tolist()
classVariable = map(int,classVariable)
testClassVariable = testDataFrame.iloc[:,-1].tolist()
testClassVariable = map(int,testClassVariable)

inputData = pd.DataFrame(inputDataFrame.iloc[:,0:inputDataFrame.shape[1]-1])
testData = pd.DataFrame(testDataFrame.iloc[:,0:testDataFrame.shape[1]-1])

trainList = []
testList = []

for eachlist in inputData.values.tolist():
    trainList.append(map(int,eachlist))

for eachlist in testData.values.tolist():
    testList.append(map(int,eachlist))

# #neural networks
# #stochastic gradient descent with 45 units in one hidden layer with linear function
solver = 'adam'
hiddenlayersizes = (45)
activation = 'relu'
start_time = time.time()
classifyUsingANN(trainList, classVariable, testList, testClassVariable,solver,hiddenlayersizes,activation)
elapsed_time = time.time() - start_time
print("Time elapsed ANN using Linear Unit with single Hidden Layer : ", elapsed_time)
#
# #stochastic gradient descent with 45 units in 1 hidden layer with tanh function
solver = 'adam'
hiddenlayersizes = (45)
activation = 'tanh'
start_time = time.time()
classifyUsingANN(trainList, classVariable, testList, testClassVariable,solver,hiddenlayersizes,activation)
elapsed_time = time.time() - start_time
print("Time elapsed ANN using Tanh Unit with single Hidden Layer : ", elapsed_time)
#
# #stochastic gradient descent with 25,20 units in 2 hidden layer with tanh function
solver = 'adam'
hiddenlayersizes = (25,20)
activation = 'tanh'
start_time = time.time()
classifyUsingANN(trainList, classVariable, testList, testClassVariable,solver,hiddenlayersizes,activation)
elapsed_time = time.time() - start_time
print("Time elapsed ANN using Tanh Unit with two Hidden Layers : ", elapsed_time)


#create different SVM classifiers
#radial basis functions with gaussian kernel
clf = createSVMClassifier('rbf')
model = "SVM using rbf kernel"
start_time = time.time()
classifyUsingSVM(trainList, classVariable, testList, testClassVariable,clf, model)
elapsed_time = time.time() - start_time
print("Time elapsed SVM using rbf Kernel : ", elapsed_time)

## linear
clf = createSVMClassifier('linear')
model = "linear SVM"
start_time = time.time()
classifyUsingSVM(trainList, classVariable, testList, testClassVariable,clf,model)
elapsed_time = time.time() - start_time
print("Time elapsed linear SVM : ", elapsed_time)

# #polynomial
clf = createSVMClassifier('poly')
start_time = time.time()
model = "SVM using poly kernel"
classifyUsingSVM(trainList, classVariable, testList, testClassVariable,clf,model)
elapsed_time = time.time() - start_time
print("Time elapsed SVM using polynomial Kernel : ", elapsed_time)

#k nearest neighbour search
KValue  = 1
start_time = time.time()
classifyUsingKNN(trainList, classVariable, testList, testClassVariable, KValue)
elapsed_time = time.time() - start_time
print("Time elapsed KNN with Kvalue 1 : ", elapsed_time)

KValue  = 10
start_time = time.time()
classifyUsingKNN(trainList, classVariable, testList, testClassVariable, KValue)
elapsed_time = time.time() - start_time
print("Time elapsed KNN with Kvalue 10 : ", elapsed_time)

KValue  = 15
start_time = time.time()
classifyUsingKNN(trainList, classVariable, testList, testClassVariable, KValue)
elapsed_time = time.time() - start_time
print("Time elapsed KNN with Kvalue 15 : ", elapsed_time)

#Gaussian Naive Bayes
start_time = time.time()
classifyUsingNB(trainList, classVariable, testList, testClassVariable)
elapsed_time = time.time() - start_time
print("Time elapsed Gaussian Naive Bayes : ", elapsed_time)

#Decision tree
start_time = time.time()
classifyUsingDecisionTree(trainList, classVariable, testList, testClassVariable)
elapsed_time = time.time() - start_time
print("Time elapsed Decision : ", elapsed_time)

#logistic regression
solver = 'sag'
multiclass = 'ovr'
start_time = time.time()
classifyUsingLogisticRegression(trainList,classVariable,testList,testClassVariable,solver,multiclass)
elapsed_time = time.time() - start_time
print("Time elapsed Logistic Regression : ", elapsed_time)


#Bayesian Network using PEBL for the data
start_time = time.time()
dataset = data.fromfile(trainFileNameForBayesian)
dataset.discretize()
learner = greedy.GreedyLearner(dataset)
ex1result = learner.run()
ex1result.tohtml("dataset")
elapsed_time = time.time() - start_time
print("Time elapsed Bayesian Network : ", elapsed_time)

