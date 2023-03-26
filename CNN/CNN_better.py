# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 15:03:08 2022

@author: pablo.sauma, edited by Josef Ruzicka
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

# Custom subdirectory to find images
DIRECTORY = "images"

def load_data():
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    names = [n.decode('utf-8') for n in unpickle(DIRECTORY+"/batches.meta")[b'label_names']]
    x_train = None
    y_train = []
    for i in range(1,6):
        data = unpickle(DIRECTORY+"/data_batch_"+str(i))
        if i>1:
            x_train = np.append(x_train, data[b'data'], axis=0)
        else:
            x_train = data[b'data']
        y_train += data[b'labels']
    data = unpickle(DIRECTORY+"/test_batch")
    x_test = data[b'data']
    y_test = data[b'labels']
    return names,x_train,y_train,x_test,y_test

names,x_train,y_train,x_test,y_test = load_data()

def plot_tensor(tensor, perm=None):
    if perm==None: perm = (1,2,0)
    plt.figure()
    plt.imshow(tensor.permute(perm).numpy().astype(np.uint8))
    plt.show()
    
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

x_train = torch.tensor(x_train,dtype=torch.float)
x_test = torch.tensor(x_test,dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

x_train = x_train.view((-1,3,32,32))
x_test = x_test.view((-1,3,32,32))

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x32x32
        self.cv1 = nn.Conv2d(3, 7, (3,3), padding='same')
        self.act1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool2d(2)
        # 7x16x16
        self.cv2 = nn.Conv2d(7, 14, (3,3), padding='same')
        self.act2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool2d(2)
        # 14x8x8
        self.cv3 = nn.Conv2d(14, 28, (3,3), padding='same')
        self.act3 = nn.LeakyReLU()
        self.pool3 = nn.MaxPool2d(2)
        # 21x4x4
        self.linear = nn.Linear(28*4*4, 128)
        self.act4 = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, 64)
        self.act5 = nn.LeakyReLU()
        self.linear3 = nn.Linear(64, 10)
        self.act6 = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.pool1(self.act1(self.cv1(x)))
        x = self.pool2(self.act2(self.cv2(x)))
        x = self.pool3(self.act3(self.cv3(x)))
        x = x.view((-1, 28*4*4))
        x = self.act4(self.linear(x))
        x = self.act5(self.linear2(x))
        return self.act6(self.linear3(x))
   
disc = CNN()
opt = optim.Adam(disc.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for i in range(250):
    opt.zero_grad()
    loss_total = 0
    for j in range(5):
        pred = disc(x_train[10000*j : 10000*(j+1)])
        loss = loss_fn(pred, y_train[10000*j : 10000*(j+1)])
        loss.backward()
        loss_total += loss.item()
    opt.step()
    with torch.no_grad():
        classification = torch.argmax(disc(x_test), dim=1)
        cf = confusion_matrix(y_test, classification)
        s = sns.heatmap(cf,xticklabels=names, yticklabels=names)
        plt.show()
        print(i, loss_total/5, sum(cf[i][i] for i in range(10))/100)
    











