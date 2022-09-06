# -- coding: utf-8 --
"""
Machine Learning, Lab 1
Josef Ruzicka, B87095.
"""

import pandas as pd 
from myPCA import myPCA

df = pd.read_csv("titanic.csv") 
pd.set_option('display.max_columns', None)

## 1) 
# Tras visualizar los datos, considero que las columnas:
# PassengerID, Ticket, Embarked y Name no son importantes con respecto a las 
# estadísticas que nos interesan, también eliminaré Cabin puesto que hay
# muchos de los valores de esta columna que se encuentran vacíos. 
# Eliminaré la columna age porque tiene algunos valores NaN, sin embargo con
# dropna(), se ignorarían esas filas, ambas alternativas a continuación.
# Eliminaré la columna Fare, ya que al centrar y reducir se llena de NaNs, no
# estoy seguro de cuál es mi error.
#df = df.drop(['PassengerId','Age','Ticket','Name', 'Cabin', 'Embarked', 'Fare'],1)
df = df.drop(['PassengerId','Ticket','Name', 'Cabin', 'Embarked', 'Fare'],1)
df = df.dropna()

# One-hot encoding
df = pd.get_dummies(df,columns=['Sex','Pclass'])

## 2)
# convertimos a matrix de numpy.
data = df.to_numpy()
    
## 3)
# We can call myPCA.analyzePrincipalComponents(data) to get C.
# however I'm calling method by method for debugging purposes.
myPCA   = myPCA()
X       = myPCA.standarizeData(data)
R       = myPCA.getCorrelationMatrix(X)
w,v     = myPCA.getEigenVectorsAndValues(R)
sw,sv   = myPCA.sortEigenVectorsAndValues(w,v)
V       = myPCA.getEigenVectorsMatrix(sv)
C       = myPCA.getPrincipalComponentsMatrix(X, V)
inertia = myPCA.getInertia(sw, C)
points  = myPCA.getCorrelationCirclePoints(V, sw)

## 4)
# Plot
import matplotlib as plt
import numpy as np
plt.pyplot.scatter(np.ravel(C[:,0]),np.ravel(C[:,1]),c = ['b' if i==1 else 'r' for i in df['Survived']])
plt.pyplot.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.pyplot.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
plt.pyplot.title('PCA')
plt.pyplot.show()

# Correlation Circle
plt.pyplot.figure(figsize=(15,15))
plt.pyplot.axhline(0, color='b')
plt.pyplot.axvline(0, color='b')
for i in range(0, df.shape[1]):
	plt.pyplot.arrow(0,0, points[0, i],  # x - PC1
              	points[1, i],  # y - PC2
              	head_width=0.05, head_length=0.05)
	plt.pyplot.text(points[0, i] + 0.05, points[1, i] + 0.05, df.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.pyplot.plot(np.cos(an), np.sin(an),color="b")  # Circle
plt.pyplot.axis('equal')
plt.pyplot.title('Correlation Circle')
plt.pyplot.show()

## 5)
# En el Scatterplot podemos observar 4 grupos, 2 de supervivientes y 2 de fallecidos pero se encuentran
# intercalados y entre cada par de grupos se podría decir que son casi linealmente divisibles por lo
# que podemos decir que sí hay ciertos (varios, por eso la intercalación de grupos) atributos con una 
# relación directa sobre la probabilidad de sobrevivir al hundimiento, en vez de ser simple aleatoriedad.

# Al observar el círculo de correlación podemos notar que la flecha más cercana a la de supervivencia
# corresponde a la de sexo femenino, esto porque se daba prioridad a las mujeres para ingresar a los botes
# salvavidas. También vemos que la clase tiene poca influencia sobre la probabilidad de sobrevivir, es probable
# que la cantidad de botes salvavidas estuviera distribuida equitativamente por el bote sin prioridad para las personas
# de cierta clase, otra explicación es que podrían haber muy pocas personas de clase alta y por esto no sea 
# una variable influyente.

## 6) 
# Si yo fuera un pasajero del Titanic, mi mejor skillset(broma) para sobrevivir consistiría en
# ser mujer y viajar en primera o segunda clase. Ser de cierta edad, así como tener una cierta cantidad de hermanos, padres o hijos
# a bordo brindarían pequeñas mejoras a mis probabilidades también.

## 7)
df = pd.read_csv("titanic.csv")
#df = df.drop(['PassengerId','Age','Ticket','Name', 'Cabin', 'Embarked'],1)
df = df.drop(['PassengerId','Ticket','Name', 'Cabin', 'Embarked', 'Fare'],1)
df = df.dropna()
df = pd.get_dummies(df,columns=['Sex','Pclass'])

# Scaling.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Correlation matrix.
from sklearn.decomposition import PCA

pca = PCA()
C = pca.fit_transform(df_scaled)

# inertia and V matrix.
inertia = pca.explained_variance_ratio_
V = pca.transform(np.identity(df_scaled.shape[1]))

# Plot
import matplotlib as plt
plt.pyplot.scatter(np.ravel(C[:,0]),np.ravel(C[:,1]),c = ['b' if i==1 else 'r' for i in df['Survived']])
plt.pyplot.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.pyplot.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
plt.pyplot.title('PCA')
plt.pyplot.show()

# Correlation Circle
plt.pyplot.figure(figsize=(15,15))
plt.pyplot.axhline(0, color='b')
plt.pyplot.axvline(0, color='b')
for i in range(0, df.shape[1]):
	plt.pyplot.arrow(0,0, V[i, 0],  # x - PC1
              	V[i, 1],  # y - PC2
              	head_width=0.05, head_length=0.05)
	plt.pyplot.text(V[i, 0] + 0.05, V[i, 1] + 0.05, df.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.pyplot.plot(np.cos(an), np.sin(an),color="b")  # Circle
plt.pyplot.axis('equal')
plt.pyplot.title('Correlation Circle')
plt.pyplot.show()

## 8)
# Ambas gráficas creadas con el uso de la biblioteca sklearn son (prácticamente)
# una versión reflejada sobre el eje x (el círculo de correlación) y sobre el eje y (el scatterplot)
# de las gráficas creadas manualmente. El resulado no se ve impactado de ninguna
# manera, la razón de esta diferencia probablemente se debe a que por algún motivo de 
# eficiencia, sklearn invierta las matrices o algún motivo similar.













