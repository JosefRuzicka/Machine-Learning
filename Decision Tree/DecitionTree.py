#Samantha Romero B87033
#Josef Ruzhicka B87095

from optparse import Values
import pandas as pd
import numpy as np
from Nodo import Nodo

class DecisionTree():
    
    def __init__(self) :
        self.root = None
        self.max_depth = None
        self.minSplitData = 2
        pass 
    
    def Gini(self, series):
        # A pandas series is equivalent to a numpy array. see https://pandas.pydata.org/docs/reference/api/pandas.Series.html
        # Gini = 1 - (sum_for_each_child(Child/propotion)^2)
        gini = 1 - (np.sum((series.value_counts()/series.shape[0])**2))
        return gini
    
    '''
    # First attempt at gini split.
    def Gini_split(self, series_array):
        # Gini split = sum_for_each_child((Child/propotion) * Gini(Child)
        gini_values = []
        series_array.dropna()
        for series in series_array:
            gini_values.append(self.Gini(series))
        gini_split = (np.sum((series_array.to_numpy()/series_array.size()) * gini_values))
        return gini_split
    '''

    def Gini_split(self, ys):
    # Slightly adapted from the web for learning purposes only. our implementation is the one commented above.
      parentNode           = ys.iloc[:, 0]
      leftChild            = ys.iloc[:, 1] 
      rightChild           = ys.iloc[:, -1]
      parentNode           = parentNode.dropna()
      leftChild            = leftChild.dropna()
      rightChild           = rightChild.dropna()
      leftChildProportion  = len(leftChild) / len(parentNode)
      rightChildProportion = len(rightChild) / len(parentNode)
      gini_split = leftChildProportion * self.Gini(leftChild) + rightChildProportion * self.Gini(rightChild)
      informacion_ganada   = self.Gini(parentNode) - gini_split 
      #print("info ganada",informacion_ganada)
      return informacion_ganada
    
    def fit(self, x, y, max_depth = None):
        self.max_depth = max_depth 
        data = pd.concat([x,y], axis=1)
        self.root = self.buildTree(data)
        print(x,y)
    
    def buildTree(self, data, my_depth =0):
        x = data.iloc[:,:-1]
        y = data.iloc[: , -1]
        dataNumber = x.shape[0]
        if self.max_depth != None:
            if (my_depth < self.max_depth and dataNumber >= self.minSplitData):
                best_split = self.getBestSplit(x, y)
                if best_split["gini"]>0:
                  
                  leftTree = self.buildTree(best_split["child-left"], my_depth+1)
                 
                  rightTree = self.buildTree(best_split["child-right"], my_depth+1)
                  
                  return Nodo(best_split["type"], best_split["count"], best_split["split-column"], 
                              best_split["split-value"], best_split["split-type"], best_split["gini"], leftTree, rightTree) 
        else:
          if (dataNumber >= self.minSplitData):
            best_split = self.getBestSplit(x, y)
            if best_split["gini"]>0:
             
              leftTree = self.buildTree(best_split["child-left"], my_depth+1)
            
              rightTree = self.buildTree(best_split["child-right"], my_depth+1)
              
              return Nodo(best_split["type"], best_split["count"], best_split["split-column"], 
                          best_split["split-value"], best_split["split-type"], best_split["gini"], leftTree, rightTree)
 
        
        leafClass = self.calculateLeaf(y)
        
        return Nodo(node_type='leaf', count=len(y), node_class=leafClass)
    
    def getBestSplit(self, x, y):
        
        bestSplitList = {}
        WinInfoMax = -float("inf")

        numericalAttribute = x.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns
        
        for attribute in x:
            values = x.loc[:, attribute]
            if attribute in numericalAttribute:
                possiblesValues, validAttribute = self.splitNumbers(values)
            else:
                possiblesValues, validAttribute = self.splitCategory(values)
            
            if(validAttribute):
                for splitValue in possiblesValues:
                    leftSons, rightSons = self.splitNodes(x, attribute, splitValue, numericalAttribute)
                    
                    if len(leftSons)>0 and len(rightSons)>0:
                        ydf, yleft, yright = y, y.loc[leftSons], y.loc[rightSons]
                        xleft, xright = x.loc[leftSons], x.loc[rightSons]
                        ys = pd.concat([ydf, yleft, yright], axis=1)
                        
                        print("YS", ys, "FIN")

                        informationGained = self.Gini_split(ys)
                       
                        if informationGained > WinInfoMax:
                          
                            dataLeft = pd.concat([xleft, yleft], axis=1)
                            dataRight= pd.concat([xright,yright], axis=1)
                            
                            bestSplitList["type"] = "split"
                            bestSplitList["count"] = len(ydf)
                            bestSplitList["split-column"] = attribute
                            bestSplitList["split-value"] = splitValue
                            if attribute in numericalAttribute:
                                bestSplitList["split-type"] = "numerical"
                            else:
                                bestSplitList["split-type"] = "categorical"
                            bestSplitList["gini"] = informationGained
                            bestSplitList["child-left"] = dataLeft
                            bestSplitList["child-right"] = dataRight
                            WinInfoMax = informationGained
        
        return bestSplitList
     
    def splitNumbers(self, attribute):
        maximunNumber = attribute.max()
        minimumNumber = attribute.min()
        
        if maximunNumber!=minimumNumber:
            intermediatePoints= np.linspace(minimumNumber , maximunNumber, num=12, retstep=True)
            ptsArray = intermediatePoints[0]
            ptsArray = ptsArray[1:11]
            return ptsArray, True
        else:
            return [], False
         
    def calculateLeaf(self, x):
      x = list(x)
      return max(x, key=x.count) 
  
    def splitCategory(self, attribute):
        categories = attribute.unique()
        
        if 1 < len(categories):
            return categories, True
        else:
            return [], False  
        
    def predict(self, df):
        predicted_classes = []
        for index,row in df.iterrows():
            currentNode = self.root
            # while current node is not a leaf.
            while (currentNode.leftChild or currentNode.rightChild):
                # Numerical node
                if (currentNode.split_type == "numerical"):
                    # left child
                    if (currentNode.leftChild and row[currentNode.split_column] <= currentNode.split_value):
                        currentNode = currentNode.leftChild
                    # right child    
                    elif (currentNode.rightChild and row[currentNode.split_column] > currentNode.split_value):
                        currentNode = currentNode.rightChild
                
                # Categorical node
                elif(currentNode.split_type == "categorical"):
                    # left child
                    if(currentNode.leftChild and row[currentNode.split_column] == currentNode.split_value):
                        currentNode = currentNode.leftChild
                    # right child
                    elif (currentNode.rightChild):
                        currentNode = currentNode.rightChild
            # End of while.
            predicted_classes.append(currentNode.node_class)      
        return pd.Series(predicted_classes)
    
    def to_dict(self, root = None):
        dictionary = {}
        currentNode = root or self.root
        print(currentNode)
        # Leaf node case
        if not (currentNode.leftChild or currentNode.rightChild):
            dictionary = {
                "type": "leaf",
                "class": currentNode.node_class,
                "count": currentNode.count
            }
        # Split node case    
        else:
            dictionary = {
                "type":         "split",
                "gini":         currentNode.gini,
                "count":        currentNode.count,
                "split-type":   currentNode.split_type,
                "split-column": currentNode.split_column,
                "split-value":  currentNode.split_value,
                "child-left":   self.to_dict(currentNode.leftChild),
                "child-right":  self.to_dict(currentNode.rightChild)
            }
        return dictionary
    
    def splitNodes(self, x, attribute, splitValue, numericaAttribute):
        
        if attribute in numericaAttribute:
          leftNumbers= x.index[x[attribute] <= splitValue]
          rightNumbers = x.index[x[attribute] > splitValue]
        else:
          leftNumbers = x.index[x[attribute] == splitValue]
          rightNumbers = x.index[x[attribute] !=  splitValue]
        return leftNumbers, rightNumbers
    
    def calculate_confusion_matrix(predict, real):
        # partially adapted from the web. 
        pclass = set(real.unique()).union(predict.unique())
        confusionMatrix = {}
        
        # Fill rows
        for i in pclass:
            currentValue = {}
            # Fill columns
            for j in pclass:
                currentValue[j] = 0
            confusionMatrix[i] = currentValue
        
        # Fill matrix. zip function combines tuples otgether. apparently works with pd.series as well.
        for p, r in zip(predict, real):
            confusionMatrix[r][p] = confusionMatrix[r][p] + 1
        return confusionMatrix 
    
