# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:46:15 2026

@author: ninuy
"""


import os
import mne

DATA_PATH = r"C:\Users\ninuy\Downloads\EEG-Motor-Imagery-Classification\Data"

file_path = os.path.join(DATA_PATH, "S001", "S001R01.edf")

raw = mne.io.read_raw_edf(file_path, preload=True)

print(raw.info)

# extract events
events, event_dict = mne.events_from_annotations(raw)

print(event_dict)
print(events[:10])

# visualize EEG
raw.plot()
