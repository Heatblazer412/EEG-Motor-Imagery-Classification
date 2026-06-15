# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:16:21 2026

@author: ninuy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Preprocessing import build_dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

X, y = build_dataset()

#Initializes X and y to seperate testing and training variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mean = X_train.mean()
std = X_train.std()

#Normalizes data specifically for training and testing data
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#maps event labels to sequential integers starting from 0, as required by cross entropy
le = LabelEncoder() 
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

#Initializing tensors specifically to document and compare model accuracy on learned vs. new data
#This step differs from our original MLP, as we do not flatten to 1 dimension yet
x_train = torch.tensor(X_train, dtype=torch.float32)
x_test = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

#We define the convolutional layer
class EEG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(64, 10)) #This step definines the parameters for the convolutional algorithm
        self.pool = nn.MaxPool2d(kernel_size=(1, 4)) #This defines the downsize parameters for the pooling step
        
        dummy = torch.zeros(1, 1, 64, X.shape[2]) #The dummy process finds the required parameter to properly map the flattened vector dimensionality
        dummy = self.pool(F.relu(self.conv1(dummy)))
        flattened_size = dummy.view(1, -1).shape[1]
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(flattened_size, 3) #This defines the dimensionality the flattened data will be configured to

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
#These are the paramaters and configurations for training the model:
learning_rate = 0.01
passes = 50
batch_size = 35

x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

dataset = TensorDataset(x_train, y_train_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = EEG_CNN() # initializes CNN object
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #this automatically finds all learnable weights in the network: conv filters, linear weights, biases 
#Training progress:
    #With a dropout rate of .1 and a single convolutional layer, this CNN achieves a peak
    #testing accuracy of roughly 61%. To achieve a higher testing accuracy we need 
    #to build more convolutional layers as to better identify relevant information for the
    #model to work with.
def Training():
    train_accuracies = []
    test_accuracies = []
    for p in range(passes):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()        # replaces W1.grad = None
            logits = model(x_batch)      # replaces manual forward pass
            loss = F.cross_entropy(logits, y_batch) 
            loss.backward()              # backpropogation
            optimizer.step()             # replaces manual weight update
        
        print(f"Pass {p+1}/{passes}, Loss: {loss.item():.4f}")    
        with torch.no_grad(): 
            logits = model(x_train)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_train_tensor).float().mean().item() * 100
            print(f"Training accuracy: {accuracy:.2f}%")
            train_accuracies.append(accuracy)
            
        with torch.no_grad(): 
            logits = model(x_test)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_test_tensor).float().mean().item() * 100
            print(f"Testing accuracy: {accuracy:.2f}%")
            test_accuracies.append(accuracy)
        
Training()