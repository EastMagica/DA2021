#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/18 13:19
# @file     : basic.py
# @project  : ML2021
# @software : Jupyter

import h5py


# Constant
# --------

wave_info = {
    'Delta': [(0.5,  4), 'tab:red'   ],
    'Theta': [(  4,  8), 'tab:orange'],
    'Alpha': [(  8, 12), 'tab:olive' ],
    'Beta' : [( 12, 35), 'tab:green' ],
    'Gamma': [( 35, 45), 'tab:blue'  ]
}


def load_data(filename):
    with h5py.File(filename, "r") as f:
        data = f['data'][...]
        emotion = f['emotion'][...]
        sampling_rate = f.attrs['sampling_rate']
    return data, emotion, sampling_rate

