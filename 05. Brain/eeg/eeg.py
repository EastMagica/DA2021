#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/18 13:10
# @file     : eeg.py
# @project  : ML2021
# @software : Jupyter

import abc

import numpy as np
import pandas as pd

from eeg.psd import psd_general
from eeg.basic import wave_info
from eeg.plot import ax_psd, EEGViewerMixIn


# Constant
# --------

psd_default_opt = {
    'name': 'periodogram',
    'fs': 128,
    'nfft': 128
}


# Classes
# -------

class EEGMeta(metaclass=abc.ABCMeta):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self._features = None
        self._columns = None


class EEG(EEGMeta, EEGViewerMixIn):
    def __init__(self, data, labels, psd_opt=dict()):
        super().__init__(data, labels)
        
        psd_opt = psd_default_opt | psd_opt
        
        self.psd = psd_general(
            x=self.data,
            **psd_opt,
        )

    @property
    def features(self):
        if self._features is None:
            self._columns, self._features = self.psd.get_features()
        return self._features
    
    def to_pandas(self, log_scale=False):
        if log_scale is True:
            features = np.log10(self.features) 
        else:
            features = self.features
        features = pd.DataFrame(
            features,
            columns=self._columns
        )
        labels = pd.Series(
            self.labels
        )
        return features, labels
