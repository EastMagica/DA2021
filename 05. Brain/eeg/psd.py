#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/17 22:48
# @file     : psd.py
# @project  : ML2021
# @software : Jupyter

import abc

import numpy as np
from scipy import signal

from eeg.basic import wave_info


# PSD Generator
# -------------

def psd_general(x, **opt):
    name = opt.pop('name')
    if name == 'periodogram':
        return Periodogram(
            x=x, 
            **opt
        )
    elif name == 'welch':
        return Welch(
            x=x, 
            **opt
        )
    else:
        msg = f"No such method :{name}"
        raise ValueError(msg)
    

# Functions
# ---------

def _find_nearest(num, array):
    return np.abs(array - num).argmin()


def _float_slice(index, array):
    patch = 0 if index.step is None else index.step
    start = None if index.start is None else _find_nearest(index.start, array)
    stop = None if index.stop is None else _find_nearest(index.stop, array) + patch
    index = slice(start, stop, None)
    return index


def _slice_process(index, array):
    """
    Processing multiply slices.
    """
    if isinstance(index, tuple) and len(index) == 2:
        index_new = (
            index[0],
            _float_slice(index[1], array)
        )
    elif isinstance(index, slice):
        index_new = _float_slice(index, array)
    elif isinstance(index, float):
        index_new = _find_nearest(index, array)
    else:
        msg = f"{self.__name__} indices must be integers or floats"
        raise TypeError(msg)
    return index_new


# Classes
# -------

class PSD(metaclass=abc.ABCMeta):
    """
    Spectral density estimation
    
    Parameters
    ----------
    x : array
        Time series of measurement values
    fs : float
        Sampling frequency of the x time series. Defaults to 1.0.
    nfft : int
        Length of the FFT used. If None the length of x will be used.
    f : array
        
    psd : array
    
    """
    def __init__(self, x, fs, nfft):
        self.x = x
        self.fs = fs
        self.nfft = nfft
        self.f = None
        self.psd = None
        
    def __len__(self):
        return self.psd.shape

    def __getitem__(self, index):
        index_new = _slice_process(index, self.f)
        if isinstance(index_new, tuple):
            f_slice = index_new[1]
        return self.f[f_slice], self.psd[index_new]
    
    def get_features(self):
        columns = list(wave_info.keys())
        features = []
        for k in columns:
            domain = wave_info[k][0]
            x, y = self[:, domain[0]:domain[1]]
            features.append(np.sum(y, axis=1))
        return columns, np.vstack(features).T
        
    @abc.abstractmethod
    def transform(self):
        raise NotImplementedError


class Periodogram(PSD):
    def __init__(self, x, fs, nfft):
        super().__init__(x, fs, nfft)
        self.transform()

    def transform(self):
        self.f, self.psd = signal.periodogram(
            x=self.x,
            fs=self.fs,
            nfft=self.nfft,
        )


class Welch(PSD):
    def __init__(self, x, fs, nfft, **opt):
        super().__init__(x, fs, nfft)
        self.transform(**opt)

    def transform(self, **opt):
        self.f, self.psd = signal.welch(
            x=self.x,
            fs=self.fs,
            nfft=self.nfft,
            **opt
        )
