# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:14:25 2026

@author: ninuy
"""

import numpy as np
from Preprocessing import build_dataset
from Preprocessing import DATA_PATH
from Preprocessing import VALID_RUNS

import os

X, y = build_dataset()

np.save(os.path.join(DATA_PATH, "X.npy"), X)
np.save(os.path.join(DATA_PATH, "y.npy"), y)
np.save(os.path.join(DATA_PATH, "valid_runs.npy"), np.array(VALID_RUNS))

print("Dataset saved.")
