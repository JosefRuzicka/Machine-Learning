# -*- coding: utf-8 -*-
'''
Lab 3 Machine Learning
Josef Ruzicka B87095
Linear Regression
'''

# Con un learning rate muy bajo (1.e-7), el error dismiuye muy 
# lentamente, y con uno muy bajo los valores se hacen NaN. 
# Probablemente se debe a un error de capa 8.
# ACTUALIZACIÓN: Creo que es por no hacer uso del threshold, 
# probablemente hay una operación aritmética ilegal en alguna época
  
# El mejor resultado se obtuvo con un lr de 1.e-5
   
# Con un train size de 0.2, los resultados son bastante 
# mejores a un test size de 0.80, no creo que sea un caso
# de overfitting así que asumo que el test size en realidad
# corresponde al tamaño de la predicción, no de los elementos a entrenar.
  
# Modificando la semilla del random_state se llega a una variación de errores
# y Scores, pero no logro ver comportamientos inusuales
   
# NOTA: Desearía haber programado el momentum y el decay, así como algún método de
# regularización, sin embargo no lo logré a tiempo, espero volver e intentarlo al
# finalizar el semestre.


from LinearRegression import *
from sklearn.model_selection import train_test_split

import pandas as pd
def main():
    
    ''' Data set loading '''
    data = pd.read_csv('fish_perch.csv')
    y_true    = data['Weight']
    y_predict = data.drop(columns= 'Weight')
    X_train, X_test, y_train, y_test = train_test_split(y_predict, y_true, test_size = 0.20, random_state=21)
    
    ''' Model test '''
    lr = LinearRegression()
    lr.fit(X_train, y_train, max_epochs=100000, threshold=1e-7, learning_rate=1e-5, 
           momentum=0, decay=0, error='mse', regularization='none', _lambda=0)
    score = lr.predict(X_test)
    print("Score: ", score)
    
    #lr.sklearn_LR(data, y_test)
    return 0
    
if __name__ == "__main__":
    main()
