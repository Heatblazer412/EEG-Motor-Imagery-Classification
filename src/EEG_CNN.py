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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#This step fetches the data from the preproceesing component 
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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(64, 10)) 
        #This step definines the parameters for the convolutional algorithm
        self.bn1 = nn.BatchNorm2d(32) 
        #This step normalizes the data from the first convolutional layer 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 10)) #Kernel size for the second convolutional layer is fitted for the output of the first
        self.bn2 = nn.BatchNorm2d(64) 
        #This step normalizes the data from the second convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=(1, 4)) 
        #This defines the downsize parameters for the pooling step
        
        dummy = torch.zeros(1, 1, 64, X.shape[2]) 
        #The dummy process finds the required parameter to properly map the flattened vector dimensionality
        dummy = self.pool(F.relu(self.conv1(dummy))) 
        #This is the dummy pass for the first layer
        dummy = self.pool(F.relu(self.conv2(dummy)))
        #This is the dummy pass for the second layer
        flattened_size = dummy.view(1, -1).shape[1]
        #This flattens the size so that it may pass through the linear network
        self.dropout = nn.Dropout(p=0.3)
        #The dropout rate is adjusted to minimize memorization (over-fitting)
        self.fc1 = nn.Linear(flattened_size, 3) 
        #This defines the dimensionality the flattened data will be configured to
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
    
#These are the paramaters and configurations for training the model:
passes = 50
batch_size = 35

x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

#This initializes the dataset and defines the train loader, processing one batch at a time
dataset = TensorDataset(x_train, y_train_tensor)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = EEG_CNN() # this initializes the CNN object
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) #this automatically finds all learnable weights in the network: conv filters, linear weights, biases 

#Training progress:
       # With a dropout rate of 0.3, learning rate of 0.0001, 50 passes,
       # and two convolutional layers, this CNN typically achieves a peak
       # testing accuracy of approximately 64%.
   # Final model evaluation:
       # Our results indicate that the network identifies class 0 (rest)
       # significantly more reliably than classes 1 and 2 (motor imagery),
       # suggesting that distinguishing motor imagery from rest is easier
       # than distinguishing between the two motor imagery tasks.
       # Analysis of the confusion matrix shows that the dominant source
       # of error is the misclassification of classes 1 and 2 as class 0.
       # The network is therefore better at detecting rest than reliably
       # identifying motor imagery activity.
def Training():
    train_accuracies = [] 
    test_accuracies = []
    best_accuracy = 0

    for p in range(passes):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()        # Optimizer replaces W1.grad = None
            logits = model(x_batch)      # Calling model on x_batch replaces our manual forward pass
            loss = F.cross_entropy(logits, y_batch)  #This calculates the loss value
            loss.backward()              # This is the backpropogation 
            optimizer.step()             # This replaces manual weight update
        
        print(f"Pass {p+1}/{passes}, Loss: {loss.item():.4f}") #This prints the loss value along with the pass number
        
        #This step finds our model's training accuracy 
        with torch.no_grad(): 
            logits = model(x_train)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_train_tensor).float().mean().item() * 100
            print(f"Training accuracy: {accuracy:.2f}%")
            train_accuracies.append(accuracy)
            
        #This step finds our testing accuracy
        with torch.no_grad(): 
            logits = model(x_test)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == y_test_tensor).float().mean().item() * 100
            print(f"Testing accuracy: {accuracy:.2f}%")
            test_accuracies.append(accuracy)
            
        #This fetches our maximum test accuracy up to this point in the loop    
        print(max(test_accuracies))
        
        #This if statement saves the best model up to this point in the loop
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
   
    # Classification Report:
        # Reports precision, recall, and F1-score for each class.
        # Precision:
            # Of all samples predicted as a class, how many were correct?
        # Recall:
            # Of all true samples belonging to a class, how many were found?
        # F1-score:
            # Harmonic mean of precision and recall.
            
    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%") #Best accuracy is printed 
    model.load_state_dict(torch.load("best_model.pth")) #Best model is loaded
    with torch.no_grad():
        logits = model(x_test)
        predictions = torch.argmax(logits, dim=1)

    print(
    classification_report(
        y_test_tensor.numpy(),
        predictions.numpy(),
        ))
    
    # Confusion Matrix:
        # Rows represent the true class.
        # Columns represent the predicted class.
        # Diagonal entries correspond to correct classifications.
        # Off-diagonal entries correspond to classification errors.
    
    cm = confusion_matrix(
    y_test_tensor.numpy(),
    predictions.numpy()
    )

    print(cm)
        
Training()
