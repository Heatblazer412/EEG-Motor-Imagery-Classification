# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:26:09 2026

@author: ninuy
"""

from Preprocessing import build_dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



#X, y fetches functionlality of Preprocessing.py
X, y = build_dataset()
print(X.shape)
print(y.shape)

input_size = X.shape[1] * X.shape[2] # the input size should be 64*321
output_size = len(np.unique(y))  # the output size should be 3
hidden_size = 16 # hidden size is configured to reduce over-fitting 

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
x_train = torch.tensor(X_train.reshape(X_train.shape[0], -1), dtype=torch.float32)
x_test = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


#This step initializes the weights and biases needed for our training algorithm:
    #W1 represents the weight configuration for the intermediate step, input -> hidden layer
W1 = torch.randn(input_size, hidden_size) * np.sqrt(1 / input_size) #Xavier initialization
W1.requires_grad_(True)
    
b1 = torch.zeros(hidden_size, requires_grad=True)
    #W2 represents the weight configuration for the final step, hidden layer -> output
W2 = torch.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
W2.requires_grad_(True)

b2 = torch.zeros(output_size, requires_grad = True)

#These are the paramaters and configurations for training the model:
learning_rate = 0.01
passes = 10
batch_size = 35
dataset = TensorDataset(x_train, y_train_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Documentation of baseline model effectiveness: 
    #Model immediately prioritizes memorization over understanding after just a couple passes 
    #Random chance baseline for 3 classes: 33.33%
    #At 1 pass a typical result will indicate a training accuracy of 54.84% & a test accuracy of 43.90%
    #At roughly 3-4 passes, we yield our optimal model resulting in, Training accuracy: 64.71%, Test accuracy: 44.97%
    #From there on, training accuracy goes up and and testing accuracy goes down
    #This overall performance indicates that a more complicated model is necessary to achieve better results, 
    #though a testing accuracy of ~45% is not insignificant, demonstrating real success in interpreting the data.
    
def Training (W1=W1, b1=b1, W2=W2, b2=b2): 
    train_accuracies = []
    test_accuracies = []
    #This function iterates over number of passes and batches in collective data
    for p in range(passes):
        for x_batch, y_batch in train_loader:
            h = torch.relu(x_batch @ W1 + b1)  #h represents the activation function, introducing non-linearity
            logits = h @ W2 + b2
            loss = F.cross_entropy(logits, y_batch) #loss is calculated, which is necessary for subsequent backpropogation
            loss.backward()
            
            with torch.no_grad(): #this step calculates gradient descent
                W1 -= learning_rate * W1.grad
                b1 -= learning_rate * b1.grad
                W2 -= learning_rate * W2.grad
                b2 -= learning_rate * b2.grad
            #This resets gradients for the next pass
            W1.grad = None
            b1.grad = None
            W2.grad = None
            b2.grad = None
            
        print(f"Pass {p+1}/{passes}, Loss: {loss.item():.4f}")
        
        #This step finds and displays our training accuracy
        with torch.no_grad():
            h = torch.relu(x_train @ W1 + b1)
            logits = h @ W2 + b2
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_train_tensor).float().mean().item() * 100
            print(f"Training accuracy: {accuracy:.2f}%")
            train_accuracies.append(accuracy)
        
        #This step finds and displays our testing accuracy
        with torch.no_grad():
            h = torch.relu(x_test @ W1 + b1)
            logits = h @ W2 + b2
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_test_tensor).float().mean().item() * 100
            print(f"Test accuracy: {accuracy:.2f}%")
            test_accuracies.append(accuracy)
            
    #Graphs training against testing accuracy
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.xlabel('Pass')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.show()

Training()
            
            
            