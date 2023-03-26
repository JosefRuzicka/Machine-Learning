"""
Created on Sat Sep 24 18:03:46 2022

@author: Josef Ruzikca B87095
ML, Lab 3, Linear Regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def MSE(y_true, y_predict):
    # Mean Squared Error
    # Error = 1 / n * ∑((f(xi) - yi)**2)
    y_predict_np = y_predict.to_numpy()
    
    # TODO: Check if instead of mean I should divide by the length of the array.
    #error = 1/y_true.shape[0] * np.subtract(y_predict_np, y_true)**2
    error = np.mean(1/y_true.shape[0] * (np.subtract(y_predict_np, y_true))**2)
    
    return error

def score(y_true_np, y_predict_np):
    # R2 version 1 = ∑ (yi - f(xi)**2) /  ∑ (yi - ∑(yi)/n )**2
    # R2 version 2 = ∑ (yi - f(xi)**2) /  ∑ (yi - mean(y))**2
    
    # TODO: check if np.sum is implicit in the subtract operation.
    numerator   = np.subtract(y_true_np, y_predict_np)**2
    denominator = np.subtract(y_true_np, np.mean(y_true_np))**2
    
    # Var name only for connoisseurs
    R2D2 = np.mean(np.divide(numerator, denominator))
    return R2D2

class LinearRegression():

    def __init__(self) :
        self.weights = None
        #self.biases  = None
        pass 
    
    def fit(self, x, y, max_epochs=100000, threshold=0.01, learning_rate=0.001,
        momentum=0, decay=0, error='mse', regularization='none', _lambda=0):
        
        ''' Set weights (C) '''
        y_predict = y.to_numpy()
        x.insert(loc=0, column = 'Bias', value=[1 for i in range(len(x.index))])
        self.weights = np.random.rand(len(x.columns))
        self.weights.shape = (self.weights.shape[0], 1)
        #self.biases  = np.ones(x[0].shape)
        #self.weights = np.concatenate(self.biases, self.weights)
        #self.weights = np.random.normal(size=len(x.columns))
        
        x = x.to_numpy()
        y_predict = y.to_numpy()
        y_predict.shape = (y_predict.shape[0], 1)
        epoch = 0        
        
        # TODO: OR change in error between last 2 epochs > thershold.
        while (epoch < max_epochs):
            
            ''' Transform data with weights. '''
            # Matmul or Dot?
            y_predict_np = np.matmul(x, self.weights)
            
            ''' Calculate current error. '''
            error = MSE(y_predict_np, y)

            ''' DEBUG error '''
            if (epoch % 10000 == 0):
                print("Epoch: ", epoch, "Error: ", error)
                
            ''' Get variable modifier. Update weights.'''
            d_error = np.dot(x.T, (y_predict_np - y_predict)) * len(1/y_predict[0])
            d_error = d_error * learning_rate
            #np.subtract(self.weights, d_error)
            self.weights = self.weights - d_error
            
            ''' TODO: add momentum, decay... calculate threshold'''
            
            epoch = epoch + 1
        return 0
    
    def predict(self, x):
        x.insert(loc=0, column = 'Bias', value=[1 for i in range(len(x.index))])
        x = x.to_numpy()
        y_predict_np = np.dot(x, self.weights)
        prediction_score = score(x, y_predict_np)
        return prediction_score
    
    def sklearn_LR(self, X, y):      
        lr = LinearRegression()
        lr.fit(X, y)
        pred = lr.predict(X)
        return 0
    