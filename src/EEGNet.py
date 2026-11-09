# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 21:15:22 2026

@author: ninuy
"""

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


DATA_PATH = r"C:\Users\ninuy\Downloads\EEG-Motor-Imagery-Classification\Data"
saved_runs = np.load(os.path.join(DATA_PATH, "valid_runs.npy"))
print(saved_runs)
#This step fetches the data from the preproceesing component 
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

import datetime
run_id = "_".join(saved_runs) + "_" + datetime.datetime.now().strftime("%H%M%S")
model_path = os.path.join(DATA_PATH, f"best_model_{run_id}.pth")

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

np.save(os.path.join(DATA_PATH, "X_test.npy"), X_test)
np.save(os.path.join(DATA_PATH, "y_test.npy"), y_test)

#We define the convolutional layer, diverging from our standard CNN by implementing two concepts:
    #Depthwise convolutions : learn spatial filters per-channel rather than as a whole
    #Separable convolutions : 
        # Depthwise step: builds a spatial aggregate across all 64 channels, independently at every timepoint,
        # compressing the per-channel temporal patterns extracted by conv1 into 2 learned spatial filters per group
        # Pointwise step: a 1x1 convolution that takes the 16 extracted feature maps and mixes them 
        # into a combined representation, far more efficiently than a single large 3D filter would
class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Depthwise step:
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(8)
        #We instantiate only eight filters, and only examine one channel at a time,
        # capturing frequency information rather than spatial, as we consider one channel across 64 timepoints
        self.depthwise = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(64, 1), groups=8)
        self.bn2 = nn.BatchNorm2d(16)
        #We build eight feature maps in consideration of the eight filters slid independently across individual channels,
        # and for every map we extract two spatial filters of information, compressing across channels into an aggregate for every time point. 
        #So we go from size [batch, 8, 64, time] in conv1 to [batch, 16, 1, time] in conv2, compressing individual channel data
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4)) 
        self.dropout1 = nn.Dropout(p=0.25)
        
        #Pointwise step:
        self.separable_depthwise = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 16), padding=(0, 8), groups=16)
        #This step extracts a new filter from each feature map, 
        # compressing across 16 timepoints of interest for the one aggregate we have of every feature map
        self.separable_pointwise = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1))
        #The final step mixes information across every filter simultaneously, at each individual timepoint
        self.bn3 = nn.BatchNorm2d(16)
        
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=0.25)
        
        #dummy pass to dynamically find the right size
        dummy = torch.zeros(1, 1, 64, X.shape[2])
        dummy = self.bn1(self.conv1(dummy))
        dummy = self.bn2(self.depthwise(dummy))
        dummy = F.elu(dummy)
        dummy = self.pool1(dummy)
        dummy = self.dropout1(dummy)
        dummy = self.separable_depthwise(dummy)
        dummy = self.separable_pointwise(dummy)
        dummy = self.bn3(dummy)
        dummy = F.elu(dummy)
        dummy = self.pool2(dummy)
        dummy = self.dropout2(dummy)
        flattened_size = dummy.view(1, -1).shape[1]
        
        self.fc1 = nn.Linear(flattened_size, 3)
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.depthwise(x))
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
      
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
      
        x = x.view(x.shape[0], -1)
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

model = EEGNet() # this initializes the CNN object
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #this automatically finds all learnable weights in the network: conv filters, linear weights, biases 

if __name__ == "__main__":
    #We define our training function
    def Training():
        train_accuracies = []  #This tracks our training accuracies
        test_accuracies = []   #This tracks our testing accuracies
        best_accuracy = 0      #This tracks our best test accuracy
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