# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:43:54 2022

@author: Josef Ruzicka B87095
Machine Learning lab 5: Convolutional Neural Network (CNN)
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.optim

# Hyper-parameters 
epochs = 3
batch_size = 4
learning_rate = 0.003

''' Use GPU, if available '''
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda')
else:
    device = ('cpu')

''' Get CIFAR10 dataset from torchvision. adapted from 'Python Engineer's PyTorch Tutorial 14''. '''
# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

''' Show Images '''
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))

class ConvNN(nn.Module):
    # Input Images: 3x32x32
    # Conv 1:       5x32x32
    # Pool:         5x16x16
    # Conv 2:       10x16x16
    # Pool:         10x8x8
    # Flatten:      640
    # linear:       output.
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, (3,3), stride=1, padding='same')
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(5, 10, (3,3), stride=1, padding='same')
        self.act2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(640, 160)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(160, 80)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(80, 10)
        self.act4 = nn.ReLU()
        self.actOut = nn.Softmax(dim=1)
            
    def forward(self, x):
        # conv 1:
        x = self.conv1(x)
        x = self.act1(x)
        
        # pool 1:
        x = self.pool(x)
        
        # conv 2:
        x = self.conv2(x)
        x = self.act2(x)
        
        # pool 2:
        x = self.pool(x)
        
        # flatten:
        x = x.view(-1, 10*8*8)
        
        # fc layers
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        
        # Output layer
        x = self.fc3(x)
        x = self.actOut(x)
        return x
        
''' Run ConvNN '''
model = ConvNN().to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Why is this not a CNN class function?
''' Training '''
for epoch in range(epochs):
    for i, (x, y_true) in enumerate(train_loader):
        x = x.to(device)
        y_true = y_true.to(device)
        
        # Forward propagation.
        y_pred = model(x)
        # Error 
        loss = criterion(y_pred, y_true)
        
        # Back propagation
        optimizer.zero_grad() # Set Δw to 0
        loss.backward()
        optimizer.step()
        
    #if (epoch % 10 == 0):
    print('epoch: ', epoch, ' loss: ', loss.item())

        
''' Store current weights '''
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)
        
# Again, why wouldn't this be a CNN class function.
''' Predict '''
# skip grad computations, since back propagation won't be executed at this point.
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    
    for x, y_true in test_loader:
        x = x.to(device)
        y_true = y_true.to(device)
        y_pred = model(x)
        
        # accuracy calculation was adapted from the tutorial mentioned earlier.
        # max returns (value ,index)
        _, predicted = torch.max(y_pred, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        
''' Plot Confusion Matrix'''
import seaborn as sns
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_true, y_pred)
sns.heatmap(CM/np.sum(CM), annot=True, fmt='.2%', cmap='Blues')

def plot_tensor(tensor, perm=None):
    if perm==None: perm = (1,2,0)
    plt.figure()            
    plt.imshow(tensor.permute(perm).numpy().astype(np.uint8))
    plt.show()

''' Load dataset alternative if folder is in directory. 
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
'''