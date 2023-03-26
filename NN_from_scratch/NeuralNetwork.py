# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 13:23:37 2022

@author: Josef Ruzicka B87095, Samantha Romero B87
Machine Learning lab 4, Neural Network
"""
import numpy as np
import math
e = 2.71828

def sigmoid(x):
    result = 1 / (1 + e**(x*-1))
    return result

def d_sigmoid(y):
    result = 1 - y**2
    return result

def tanh(x):
    result = (e**x - e**(x*-1))/(e**x + e**(x*-1))
    return result

def d_tanh(y):
    result = 1 - y**2
    return result

def relu(x):
    result = 0
    if (x > 0):
        result = x
    return result

def d_relu(y):
    result = 0
    if (y > 0):
        result = 1
    return result

def lrelu(x):
    result = x
    if (x <= 0):
        result *= 0.01
    return result

def d_lrelu(y):
    result = 1
    if (y <= 0):
        result = 0.01
    return result

def MSE(prediction, expected):
    # Adapted from geeks for geeks.
    error = np.square(np.subtract(expected, prediction)).mean()
    return error

def d_MSE(prediction, expected):
    d_error = (2 * np.subtract(expected, prediction)).mean()
    return d_error

class DenseNN(): 
    
    layers        = []
    weights       = []
    activation    = []
    layerOutputs  = []
    lr            = 0
    epoch         = 0
    delta_weights = []
    momentum      = 0
    decay         = 0
    errors        = []

    def __init__(self, layers, activation, seed=0):
        
        '''Xavier initialization'''
        for i in range(len(layers)-1):
            # start i in 1 becausecause there are no weights before input layer.
            i += 1
            fan_in = layers[i-1]
            fan_out = layers[i+1]
            
            # loc = mean, scale = std
            np.random.seed(seed)
            self.weights[i] = np.random.normal(loc=0, scale=2.0/math.sqrt([fan_in + fan_out]), size=None)
            #TODO: ADD BIAS.
            
            i -= 1
            
        '''Data Storing'''
        self.activation = activation
        self.layers = layers
            
    def predict(self, x):
        output = self.forwardpropagation(x)
        return output

    def forwardpropagation(self, x):
        # Activation(Activation(data * weights[0]) * weights[1])
        # TODO: Assert x columns == layer neuron count.
        previousLayerOutput = x
        currentLayerOutput = 0
        for i in range(len(self.layers)):
            currentLayerOutput = previousLayerOutput * self.weights[i-1] 
            
            ''' Select Activation function.'''
            #TODO: Create a separate function for the following code
            if (self.activation[i] == 's'):
                currentLayerOutput = sigmoid(currentLayerOutput)
            elif (self.activation[i] == 't'):
                currentLayerOutput = tanh(currentLayerOutput)
            elif (self.activation[i] == 'r'):
                currentLayerOutput = relu(currentLayerOutput)
            elif (self.activation[i] == 'l'):
                currentLayerOutput = lrelu(currentLayerOutput)
            
            # Update previous layer result.
            self.layerOutputs[i] = currentLayerOutput
            previousLayerOutput  = currentLayerOutput
            
        # Todo: Threshold on output layer result?
        return currentLayerOutput
    
    def backpropagation(self, x, y):

        ''' Get output layer's error'''
        #error   = MSE(x, y)
        d_error = d_MSE(x, y)
        d_activation = 0
        
        # might need to sum 1 to the len for the reversed to work.
        # TODO: Use numpy?
        for i in reversed(range(len(self.layers))):
            
            ''' Select Activation function. '''
            #TODO: Create a separate function for the following code
            if (self.activation[i] == 's'):
                d_activation = d_sigmoid(self.layerOutputs[i])
            elif (self.activation[i] == 't'):
                d_activation = d_tanh(self.layerOutputs[i])
            elif (self.activation[i] == 'r'):
                d_activation = d_relu(self.layerOutputs[i])
            elif (self.activation[i] == 'l'):
                d_activation = d_lrelu(self.layerOutputs[i])
            
            ''' Update delta weights '''
            # * layerOutput[i] ?
            gradient = d_error * d_activation
            # delta_W = output_K * gradient_K+1
            self.delta_weights[i-1] = self.layerOutputs[i-1] * gradient
            
            # move to step:
            #self.weights[i] -= self.lr * gradient
            
            ''' Set up error for next weights (previous layer) '''
            # TODO: check if formulae is correct 
            d_error = self.weights[i-1] * gradient
            #d_error = self.delta_weights[i] * gradient
            
    def train(self, lr=0.05, momentum=0, decay=0):
        self.delta_weights = []
        self.epoch = 0
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        
        # TODO: get x and y
        x = [(0,0),(0,1),(1,0),(1,1)]
        y = [0,1,1,0]
        
        # TODO: epochs?
        while self.epoch < 100:

            self.forwardpropagation(x)
            ''' Get error '''
            error = MSE(x, y)
            self.errors.append(error)
            self.backpropagation(x, y)
            self.step()
            
            
    def step(self):
        ''' Update weights '''
        # TODO: use numpy ?
        self.weights -= self.delta_weights * self.lr
        
        ''' Update lr, and momentum '''
        self.lr = self.lr / 1 + self.decay
        # self.momentum = TODO
         
        self.epoch += 1
        
    
        
        
        
        
        