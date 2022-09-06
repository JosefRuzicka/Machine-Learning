# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 20:46:14 2022

@author: Josef Ruzicka
"""
import math
import numpy as np
from numpy.linalg import eig

class myPCA:
  
    def standarizeData(self, dataMatrix):
        result = dataMatrix      
        mean = np.mean(dataMatrix, axis=0)
        std = np.std(dataMatrix, axis=0)
        result = (dataMatrix - mean) / std
        return result
    
    def getCorrelationMatrix(self, standarizedMatrix):
        correlationMatrix = (1/len(standarizedMatrix))*(np.transpose(np.copy(standarizedMatrix))).dot(standarizedMatrix)
        return correlationMatrix
    
    def getEigenVectorsAndValues(self, correlationMatrix):
        w, v = eig(correlationMatrix)
        return w, v
    
    def sortEigenVectorsAndValues(self, eigenValues, eigenVectors):
        # https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors

    def getEigenVectorsMatrix(self, eigenVectors):
        matrix = np.array(eigenVectors)
        return matrix
    
    def getPrincipalComponentsMatrix(self, dataMatrix, eigenVectorsMatrix):
        matrix = np.matmul(dataMatrix,eigenVectorsMatrix)
        return matrix
    
    def analyzePrincipalComponents(self, dataMatrix):
        X       = self.standarizeData(dataMatrix)
        R       = self.getCorrelationMatrix(X)
        w,v     = self.getEigenVectorsAndValues(R)
        sw,sv   = self.sortEigenVectorsAndValues(w,v)
        V       = self.getEigenVectorsMatrix(v)
        C       = self.getPrincipalComponentsMatrix(X, V)
        return C
    
    def getInertia(self, eigenValues, principalComponentsMatrix):
        inertia = eigenValues / len(principalComponentsMatrix[0])
        return inertia

    def getCorrelationCirclePoints(self, eigenVectorsMatrix, eigenValues):
        points = (eigenVectorsMatrix[:,0] * math.sqrt(eigenValues[0])), (eigenVectorsMatrix[:,1] * math.sqrt(eigenValues[1]))  
        return np.array(points)