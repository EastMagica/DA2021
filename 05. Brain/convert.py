#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/18 20:28
# @file     : plot.py
# @project  : ML2021
# @software : Jupyter

import h5py
import numpy as np
import pandas as pd


# Convert Data to HDF5
data = pd.read_excel("eeg_emotion.xlsx")

with h5py.File("eeg_emotion.hdf5", "w") as f:
    f['features'] = data.iloc[:, :-2].to_numpy(dtype=np.float64)
    f['labels'] = data.iloc[:, -2:].to_numpy(dtype=np.float64)
    f.attrs['name'] = 'EEG Emotion'
    f.attrs['sampling_rate'] = 120
    f.attrs['time'] = 60


# # Add new Data
# with h5py.File("eeg_emotion.hdf5", "a") as f:
#     f['psd'] = psd
#     f['category'] = category


# # get all keys
# with h5py.File("eeg_emotion.hdf5", "r") as f:
#     print(f.keys())
#     print(f.attrs.keys())
