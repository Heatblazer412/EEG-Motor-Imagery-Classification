# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:06:51 2026

@author: ninuy
"""
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from EEG_CNN import EEG_CNN
from EEGNet import EEGNet
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


#ENSEMBLE ANALYSIS

# ==========================================================
# Fetch Models + Dataset
# ==========================================================

DATA_PATH = r"C:\Users\ninuy\Downloads\EEG-Motor-Imagery-Classification\Data"
batch_size = 35
X = np.load(os.path.join(DATA_PATH, "X.npy"))
y = np.load(os.path.join(DATA_PATH, "y.npy"))

# ==========================================================
# Model Paths
# ==========================================================

#RAW CNN MODELS:
raw_model_path = os.path.join(
    DATA_PATH,
    "raw_model_R03_R07_R11_095403.pth"
)

raw_model_path_2 = os.path.join(
    DATA_PATH,
    "raw_model_R03_R07_R11_120501.pth"
)

raw_model_path_3 = os.path.join(
    DATA_PATH,
    "raw_model_R03_R07_R11_121716.pth"
)

#EEGNet MODELS:

eegnet_model_path = os.path.join(
    DATA_PATH,
    "best_model_R03_R07_R11_165033.pth"
)

eegnet_model_path_2 = os.path.join(
    DATA_PATH,
    "best_model_R03_R07_R11_081540.pth"
)

eegnet_model_path_3 = os.path.join(
    DATA_PATH,
    "best_model_R03_R07_R11_164307.pth"
)

# Instantiate Model Objects:
raw_model_1 = EEG_CNN(64)
raw_model_2 = EEG_CNN(64)
raw_model_3 = EEG_CNN(64)

csp_model = EEG_CNN(20)

eegnet_model_1 = EEGNet()
eegnet_model_2 = EEGNet()

# ==========================================================
# Load Weights Within Model Objects
# ==========================================================

#LOADING CNN:
raw_model_1.load_state_dict(
    torch.load(raw_model_path)
)

raw_model_2.load_state_dict(
    torch.load(raw_model_path_2)
)

raw_model_3.load_state_dict(
    torch.load(raw_model_path_3)
)

#LOADING EEGNet:
eegnet_model_1.load_state_dict(
    torch.load(eegnet_model_path)
)

eegnet_model_2.load_state_dict(
    torch.load(eegnet_model_path_2)
)

# ==========================================================
# Evaluation Mode
# ==========================================================

raw_model_1.eval()
raw_model_2.eval()
raw_model_3.eval()

csp_model.eval()

eegnet_model_1.eval()
eegnet_model_2.eval()

X_test = np.load(os.path.join(DATA_PATH, "X_test.npy"))
y_test = np.load(os.path.join(DATA_PATH, "y_test.npy"))

X_test_csp = np.load(os.path.join(DATA_PATH, "X_test_csp.npy"))
y_test_csp = np.load(os.path.join(DATA_PATH, "y_test_csp.npy"))

x_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
x_test_csp = torch.tensor(X_test_csp, dtype=torch.float32).unsqueeze(1)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)
# ==========================================================
# Analysis Functions
# ==========================================================

# Functions here evaluate model performance independently.
def get_model_predictions(model, x):
    model.eval()

    with torch.no_grad():
        logits = model(x)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    return predictions, probabilities

#Ensembles a collection of evaluations
def ensemble_predictions(probability_list):
    
    ensemble_probabilities = sum(probability_list) / len(probability_list)

    ensemble_predictions = torch.argmax(
        ensemble_probabilities,
        dim=1
    )

    return ensemble_predictions, ensemble_probabilities
    
# Raw CNN predictions
_, raw_prob_1 = get_model_predictions(raw_model_1, x_test)
_, raw_prob_2 = get_model_predictions(raw_model_2, x_test)
_, raw_prob_3 = get_model_predictions(raw_model_3, x_test)


# EEGNet predictions
_, eegnet_prob_1 = get_model_predictions(eegnet_model_1, x_test)
_, eegnet_prob_2 = get_model_predictions(eegnet_model_2, x_test)

#Execute Ensemble:
ensemble_pred, ensemble_prob = ensemble_predictions(
    [
        eegnet_prob_1,
        eegnet_prob_2,
    ]
)

#Show Results:
print(ensemble_pred)
accuracy = (
    ensemble_pred == y_test_tensor
).float().mean().item()
print(f"Ensemble Accuracy: {accuracy*100:.2f}%")

