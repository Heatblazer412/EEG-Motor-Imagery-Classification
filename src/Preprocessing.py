# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:46:15 2026

@author: ninuy
"""

import os
import mne
import numpy as np


DATA_PATH = r"C:\Users\ninuy\Downloads\EEG-Motor-Imagery-Classification\Data"
VALID_RUNS = ["R04", "R08", "R12"]

file_path = os.path.join(DATA_PATH, "S001", "S001R04.edf")

#for each .edf file, this function loads the data, filters for relevant bandwith,
#before extracting event labels and assigning them to individual epochs within the data.
def load_and_preprocess(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True) #loads raw data
    filtered = raw.copy().filter(8, 30) #filters for bandwith

    events, event_dict = mne.events_from_annotations(filtered) #extracts event labels

    epochs = mne.Epochs( #builds epochs
        filtered,
        events,
        event_id=event_dict,
        tmin=0,
        tmax=2,
        baseline=None,
        preload=True
    )

    X = epochs.get_data() #extracts epochs 
    y = epochs.events[:, -1] #extracts individual event labels
    
    return X, y

#this function builds the overall dataset using, load_and_preprocess(), so we may have
#an accumulated data set relevent to this specific use of our event labels
def build_dataset():
    #initializing structure to collect epochs/event labels
    all_X = [] 
    all_y = []

    for subject in os.listdir(DATA_PATH): #iterates over all individuals

        subject_path = os.path.join(DATA_PATH, subject) #selects subject path, before filtering over valid trials

        if not os.path.isdir(subject_path):
            continue

        for file in os.listdir(subject_path):

            if not file.endswith(".edf"):
                continue

            if not any(run in file for run in VALID_RUNS):
                continue

            file_path = os.path.join(subject_path, file) 

            print(f"Processing: {file}")

            X, y = load_and_preprocess(file_path) #loads and preprocesses data for individual trial

            all_X.append(X)
            all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    return X, y

build_dataset()
