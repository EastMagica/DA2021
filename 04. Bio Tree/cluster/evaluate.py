#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/13 20:42
# @file     : evaluate.py
# @project  : ML2021
# @software : Jupyter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cluster.cluster import Cluster
from cluster.rfdist import split_list_tree, rf_distance
from cluster.basic import standarded_tree, get_weighted_hamming


# Constant
# --------

standarded_split_tree = split_list_tree(standarded_tree)


# Functions
# ---------

def cluster_all(data, imputer=None, name='', show=True, save=True, form='png', figsize=(10, 8)):
    method = ['single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward']
    metric = ['hamming', 'hamming', 'hamming', 'hamming', 'euclidean', 'euclidean', 'euclidean']

    method_names = ["Standarded"] + [f"{method[i]}({metric[i]})" for i in range(7)]
    
    method_rate_mat = np.zeros((8, 8), dtype=float)
    
    method_rate_mat[0, 0] = rf_distance(
        standarded_split_tree, 
        standarded_split_tree, 
        (len(standarded_split_tree[0]) - 1) // 2
    )

    for i in range(1, 8):
        c1 = Cluster(
            data,
            method=method[i-1],
            metric=metric[i-1],
            imputer=imputer
        )
        method_rate_mat[0, i] = c1.rf_distance(standarded_split_tree)
        for j in range(1, 8):
            if i <= j:
                c2 = Cluster(
                    data,
                    method=method[j-1],
                    metric=metric[j-1],
                    imputer=imputer
                )
                method_rate_mat[i, j] = c1.rf_distance(c2)
    
    rate_mate = pd.DataFrame(method_rate_mat + method_rate_mat.T, index=method_names, columns=method_names)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        rate_mate,
        ax=ax,
        annot=True,
        fmt='.2f',
        center=0,
        cmap="Spectral",
    )
    
    # ax.tick_params(axis='x', rotation=30, ha="right")
    plt.setp(
        ax.get_xticklabels(), 
        rotation=30, 
        ha="right",
        rotation_mode="anchor",
        fontsize=15,
    )
    plt.setp(
        ax.get_yticklabels(),
        fontsize=15
    )

    if save is True:
        plt.savefig(f"{name}_cluster_mat.{form}", dpi=120)
    if show is True:
        plt.show()
    plt.clf()
    
    return rate_mate


# def get_
    
    