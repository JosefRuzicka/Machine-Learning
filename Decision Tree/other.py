#Samantha Romero B87033
#Josef Ruzhicka B87095

# Desition Tree, ML Lab 2

import DecisionTree as Dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree

def loadDataMushrooms():
    #Import dataframe mushrooms
    dfMushrooms = pd.read_csv ("Content\\mushrooms.csv")
    yMushrooms = dfMushrooms['class']
    xMushrooms = dfMushrooms.loc[ : , dfMushrooms.columns != 'class']
    return (xMushrooms, yMushrooms)

#Import dataframe Iris   
def loadDataIris():
    dfIris = pd.read_csv ("Content\\iris.csv")
    dfIris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "Target"]
    
    yIris = dfIris['Target']
    xIris = dfIris.loc[ : , dfIris.columns != 'Target']
    return (xIris, yIris)

#Import dataframe Titanic
def loadDataTitanic():
    dfTitanic = pd.read_csv ("Content\\titanic.csv")
    yTitanic = dfTitanic['Survived']
    xTitanic = dfTitanic.loc[ : , dfTitanic.columns != 'Survived']
    
    return (xTitanic, yTitanic)

class main: 
    dfTitanic = loadDataTitanic()
    dfIrisX, dfIrisY = loadDataIris()
    dfMushrooms = loadDataMushrooms()
   
    #Choose dataset
    dataset1 = dfIrisX
    dataset2 = dfIrisY
    
    xtrain, xtest, ytrain, ytest = train_test_split(dataset1, dataset2, test_size=.2, random_state=41)
    X_train, X_test, y_train, y_test = train_test_split(dataset1, dataset2, train_size=0.75, test_size=0.25, random_state=0)

    #Testing train tree
    decisionT = Dt.DecisionTree()
    decisionT.fit(xtrain,ytrain, max_depth=3)
    tree = decisionT.to_dict()
    print(tree)
    
    tree = DecisionTreeClassifier(criterion='gini', random_state=0, max_depth = 3)
    tree.fit(X_train,y_train)
    prediction = tree.predict(X_test)
    # Y para graficar
    fig = plt.figure(figsize=(25,20))
    _ = plot_tree(tree, feature_names=X_train.columns,class_names=["Died", "Survived"], filled=True)
    plt.show()
