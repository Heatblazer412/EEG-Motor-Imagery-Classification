# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:16:21 2026

@author: ninuy
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import datetime

# ==========================================================
# Configuration and Loading
# ==========================================================

#These are the paramaters and configurations for training the model:
CSP_RUN=True
passes = 50
batch_size = 35

DATA_PATH = r"C:\Users\ninuy\Downloads\EEG-Motor-Imagery-Classification\Data"
saved_runs = np.load(os.path.join(DATA_PATH, "valid_runs.npy"))
print(saved_runs)

import datetime
run_id = "_".join(saved_runs) + "_" + datetime.datetime.now().strftime("%H%M%S")
model_path = os.path.join(DATA_PATH, f"best_model_{run_id}.pth")

#This step fetches the data from the preproceesing component 
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))
le = LabelEncoder()
y = le.fit_transform(y)
print(X.shape)
print(np.unique(y, return_counts=True))

# ==========================================================
# CNN Defenition
# ==========================================================
#We define the convolutional layer
class EEG_CNN(nn.Module):
    def __init__(self, n_channels):
        super().__init__()      
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(n_channels, 10)) 
        #This step definines the parameters for the convolutional algorithm
        self.bn1 = nn.BatchNorm2d(32) 
        #This step normalizes the data from the first convolutional layer 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 10)) #Kernel size for the second convolutional layer is fitted for the output of the first
        self.bn2 = nn.BatchNorm2d(64) 
        #This step normalizes the data from the second convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=(1, 4)) 
        #This defines the downsize parameters for the pooling step
        
        dummy = torch.zeros(1, 1, n_channels, X.shape[2]) #The dummy process finds the required parameter to properly map the flattened vector dimensionality
        dummy = self.pool(F.relu(self.conv1(dummy))) 
        #This is the dummy pass for the first layer
        dummy = self.pool(F.relu(self.conv2(dummy)))
        #This is the dummy pass for the second layer
        flattened_size = dummy.view(1, -1).shape[1]
        #This flattens the size so that it may pass through the linear network
        self.dropout = nn.Dropout(p=0.5)
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
    

# ==========================================================
# Raw CNN Dataset
# ==========================================================

#Initializes X and y to seperate testing and training variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mean = X_train.mean()
std = X_train.std()

#Normalizes data specifically for training and testing data
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

n_channels = X_train.shape[1]

#maps event labels to sequential integers starting from 0, as required by cross entropy
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

#Initializing tensors specifically to document and compare model accuracy on learned vs. new data
#This step differs from our original MLP, as we do not flatten to 1 dimension yet
x_train = torch.tensor(X_train, dtype=torch.float32)
x_test = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

x_train = x_train.unsqueeze(1)
x_test = x_test.unsqueeze(1)

raw_model = EEG_CNN(X_train.shape[1])
raw_dataset = TensorDataset(x_train, y_train_tensor)
raw_train_loader = DataLoader(
    raw_dataset,
    batch_size=batch_size,
    shuffle=True
)



raw_optimizer = torch.optim.Adam(
    raw_model.parameters(),
    lr=0.0001
)
raw_model_path = os.path.join(
    DATA_PATH,
    f"raw_model_{run_id}.pth"
)
# ==========================================================
# CSP Dataset
# ==========================================================
# Introducing CSP data processing. CSP (Common Spatial Patterns) takes information present in the raw
# EEG scan and learns spatial associations that maximally differentiate activity between regions of interest.
# For example, in the case of L/R motor imagery it excagerates data in either hemisphere of the brain.

from mne.decoding import CSP
if (CSP_RUN):
    csp = CSP(
            n_components=20,
            reg='ledoit_wolf',
            log=True,
            norm_trace=False
            )
    #This filters epochs to only left vs right (exclude rest)
    mask = (y == 1) | (y == 2)
    X_motor = X[mask]
    y_motor = y[mask]

    le_binary = LabelEncoder()
    y_motor = le_binary.fit_transform(y_motor)
    
    #This fits CSP on left vs right only
    csp.fit(X_motor, y_motor)

    # We bypass transform(), Projecting all epochs (including rest) manually.
    # X shape: (Epochs, 64, 320)
    # csp.filters_ shape: (8, 64)
    filters = csp.filters_[:20]
    X_csp = np.matmul(filters, X_motor)
    X_csp = (X_csp - np.mean(X_csp, axis=-1, keepdims=True)) / np.std(X_csp, axis=-1, keepdims=True)
    
    # Verify the new shape
    n_channels = X_csp.shape[1]
    print(X_csp.shape) # Expected output: (Epochs, 8, 320)
    
    X_train_csp, X_test_csp, y_train_csp, y_test_csp = train_test_split(
        X_csp,
        y_motor,
        test_size=0.2,
        random_state=42,
        stratify=y_motor
    )

    mean = X_train_csp.mean()
    std = X_train_csp.std()
    
    X_train_csp = (X_train_csp - mean)/std
    X_test_csp = (X_test_csp - mean)/std
    
    x_train_csp = torch.tensor(
        X_train_csp,
        dtype=torch.float32
    ).unsqueeze(1)
    
    x_test_csp = torch.tensor(
        X_test_csp,
        dtype=torch.float32
    ).unsqueeze(1)
    
    print(np.unique(y_train_csp))
    print(np.unique(y_test_csp))
    
    y_train_csp_tensor = torch.tensor(
        y_train_csp,
        dtype=torch.long
    )
    
    y_test_csp_tensor = torch.tensor(
        y_test_csp,
        dtype=torch.long
    )
    
    csp_dataset = TensorDataset(
        x_train_csp,
        y_train_csp_tensor
    )
    
    csp_model = EEG_CNN(X_train_csp.shape[1])
    csp_train_loader = DataLoader(
        csp_dataset,
        batch_size=batch_size,
        shuffle=True
    )    
    
    
    csp_optimizer = torch.optim.Adam(
        csp_model.parameters(),
        lr=0.001
    )
    
    
    csp_model_path = os.path.join(
        DATA_PATH,
        f"csp_model_{run_id}.pth"
    )
    print(np.unique(y))
    print(np.unique(y_train))
    print(np.unique(y_train_csp))
    
#We define our training function
def Training(
    model,
    optimizer,
    train_loader,
    x_train,
    y_train_tensor,
    x_test,
    y_test_tensor,
    model_path,
    model_name
):
    
    train_accuracies = []  #This tracks our training accuracies
    test_accuracies = []   #This tracks our testing accuracies
    best_accuracy = 0      #This tracks our best test accuracy
    print(f"\n========== {model_name} ==========")
    print(saved_runs)      #This prints the last saved Valid Run input for Preprocessing.py

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
            torch.save(model.state_dict(), model_path)
            
   
    # Classification Report:
        # Reports precision, recall, and F1-score for each class.
        # Precision:
            # Of all samples predicted as a class, how many were correct?
        # Recall:
            # Of all true samples belonging to a class, how many were found?
        # F1-score:
            # Harmonic mean of precision and recall.
            
    print(f"\nBest Test Accuracy: {best_accuracy:.2f}%") #Best accuracy is printed 
    model.load_state_dict(torch.load(model_path)) #Best model is loaded
    model.eval()
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
        
Training(
    csp_model,
    csp_optimizer,
    csp_train_loader,
    x_train_csp,
    y_train_csp_tensor,
    x_test_csp,
    y_test_csp_tensor,
    csp_model_path,
    "CSP CNN"
)    
    
Training(
    raw_model,
    raw_optimizer,
    raw_train_loader,
    x_train,
    y_train_tensor,
    x_test,
    y_test_tensor,
    raw_model_path,
    "Raw CNN"
)

