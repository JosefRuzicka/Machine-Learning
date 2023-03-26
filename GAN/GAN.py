# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:37:01 2022

@author: Josef Ruzicka B87095
Machine Learning, Lab 6 GAN.

Useful references:
https://towardsdatascience.com/what-are-transposed-convolutions-2d43ac1a0771
https://www.cs.toronto.edu/~lczhang/360/lec/w09/gan.html
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
epochs = 2
learning_rate = 0.003

''' Use GPU, if available '''
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda')
else:
    device = ('cpu')

''' Get MNIST data set '''
train = torchvision.datasets.MNIST(".", download=True)
x = train.data.float()
y = train.targets

class Classifier(nn.Module):
    # Input Images: 1x28x28
    # Conv 1:       5x28x28
    # Pool:         5x14x14
    # Conv 2:       10x14x14
    # Pool:         10x7x7
    # Flatten:      490
    # linear:       output.
    def __init__(self):
        # TODO: use self.model = nn.sequential to simplify forward.
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, (3,3), stride=1, padding='same')
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(5, 10, (3,3), stride=1, padding='same')
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.fc1  = nn.Linear(490, 200)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.fc2  = nn.Linear(200, 50)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)
        self.fc3  = nn.Linear(100, 10)
        self.act5 = nn.LeakyReLU(0.2, inplace=True)
        self.fc4  = nn.Linear(10, 2)
        self.actOut = nn.Softmax(dim=1)
            
    def forward(self, x):
        # conv 1 and pool:
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool(x)
        
        # conv 2 and pool:
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool(x)
        
        # flatten:
        x = x.view(-1, 10*7*7)
        
        # fc layers
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)
        x = self.act5(x)
        
        # Output layer
        x = self.fc4(x)
        x = self.actOut(x)
        return x
    
    def train_classifier(opt, model, x_true, x_false, accuracy=None, max_iters=100, batch_size=1000):
        for iter in range(max_iters):
            x_pred = model(x)
            loss = criterion(x_pred, x_true)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return 0
    
class Generator(nn.Module): 
    # Input Images: 490 array of random numbers (noise).
    # reshape (unflatten?): 10x7x7
    # transposedConv 1:     5x7x7
    # unPool:               5x14x14
    # transposedConv 2:     1x14x14
    # unPool:               1x28x28
    
    def __init__(self):
        # TODO: BatchNorm2d instead of unpooling?
        super(Generator, self).__init__()
        self.transposed_conv1 = nn.ConvTranspose2d(10,  5, (3,3), stride=1)
        self.unpool1 = nn.MaxUnpool2d(2)
        self.transposed_conv2 = nn.ConvTranspose2d(5, 1, (3,3), stride=1)
        self.unpool2 = nn.MaxUnpool2d(2)
        self.actConvAndLinnear = nn.ReLU()
            
    def forward(self, x):

        # conv 1 and pool:
        x = self.transposed_conv1(x)
        x = self.unpool1(x)
        x = self.transposed_conv2(x)
        x = self.unpool2(x)
        
        # conv 2 and pool:
        x = self.conv2(x)
        x = self.actConvAndLinnear(x)
        x = self.unpool(x)

        return x
    
    def train_generator(opt, generator, classifier, accuracy=None, max_iters=100, batch_size=1000):
        x = torch.randn(490)
        x = torch.reshape(x, (10,7,7))
        '''
        for i, (x, y_true) in enumerate(train_loader):
            y_true = model(x) 
            
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
            '''
        return 

def plot_image(img,save=False,name=None):
    cmap = 'gray' if img.shape[0]==1 else None
    data = (img.detach()*(255  if img.max()<=1 else 1)).permute((1,2,0)).numpy().astype(np.uint8)
    plt.figure()
    plt.imshow(data,cmap=cmap)
    plt.show()
    if save:
        if img.shape[0]==1:
            plt.imsave(name, data.squeeze(), cmap=cmap)
        else:
            plt.imsave(name, data, cmap=cmap)
            
''' Main '''
generator = Generator()
classifier = Classifier()            
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)   
#criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()

#for epoch in range(epochs):
    # Train Generator
    
    # Train Classifier/Discriminator


''' BELOW THIS POINT THE CODE WAS TAKEN FROM https://www.cs.toronto.edu/~lczhang/360/lec/w09/gan.html'''
''' IN THE NEAR FUTURE ILL CODE MY OWN VERSION AFTER LEARNING FROM THIS ONE '''
''' KEEP IT SECRET, KEEP IT SAFE (from the lads at toronto uni) -GANDALF THE GREY '''
import torch.utils.data
from torchvision import datasets

mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
def train(generator, discriminator, lr=0.001, num_epochs=5):
    criterion = nn.BCEWithLogitsLoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=100, shuffle=True)

    num_test_samples = 16
    test_noise = torch.randn(num_test_samples, 100)

    for epoch in range(num_epochs):
        # label that we are using both models 
        generator.train()
        discriminator.train()


        for n, (images, _) in enumerate(train_loader):
            # === Train the Discriminator ===

            noise = torch.randn(images.size(0), 100)
            fake_images = generator(noise)
            inputs = torch.cat([images, fake_images])
            labels = torch.cat([torch.zeros(images.size(0)), # real
                                torch.ones(images.size(0))]) # fake

            d_outputs = discriminator(inputs)
            d_loss = criterion(d_outputs, labels)
            d_loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()

            # === Train the Generator ===
            noise = torch.randn(images.size(0), 100)
            fake_images = generator(noise)
            outputs = discriminator(fake_images)

            g_loss = criterion(outputs, torch.zeros(images.size(0)))
            g_loss.backward()
            g_optimizer.step()
            g_optimizer.zero_grad()

        scores = torch.sigmoid(d_outputs)
        real_score = scores[:images.size(0)].data.mean()
        fake_score = scores[images.size(0):].data.mean()


        print('Epoch [%d/%d], d_loss: %.4f, g_loss: %.4f, ' 
              'D(x): %.2f, D(G(z)): %.2f' 
              % (epoch + 1, num_epochs, d_loss.item(), g_loss.item(), real_score, fake_score))
        
        # plot images
        generator.eval()
        discriminator.eval()
        test_images = generator(test_noise)
        plt.figure(figsize=(9, 3))
        for k in range(16):
            plt.subplot(2, 8, k+1)
            plt.imshow(test_images[k,:].data.numpy().reshape(28, 28), cmap='Greys')
        plt.show()

discriminator = Classifier()
generator = Generator()
train(generator, discriminator, lr=0.001, num_epochs=20)
