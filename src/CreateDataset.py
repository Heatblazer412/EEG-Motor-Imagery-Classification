# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 16:14:25 2026

@author: ninuy
"""

import numpy as np
from Preprocessing import build_dataset
from Preprocessing import DATA_PATH
import os

X, y = build_dataset()

np.save(os.path.join(DATA_PATH, "X.npy"), X)
np.save(os.path.join(DATA_PATH, "y.npy"), y)

print("Dataset saved.")