#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/13 15:26
# @file     : cluster.py
# @project  : ML2021
# @software : Jupyter

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

import matplotlib.pyplot as plt
import seaborn as sns


# init
# ----

sns.set_theme()

cmap = 'vlag'  # vlag, Spectral, coolwarm


# Functions
# ---------

def get_weighted_hamming(w):
    def weighted_hamming(u, v):
        return np.sum((u == v).astype(float) * w)
    return weigthed_hamming


def show_heatmap(data, title='', name='', show=True, save=False, form='png', figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    sns.heatmap(
        data,
        ax=ax,
        center=0,
        cmap=cmap,
        # linewidths=0.1,
        xticklabels=True,
        yticklabels=True,
    )
    if save is True:
        plt.savefig(f"{self.name}_heatmap.{form}", dpi=120)
    if show is True:
        plt.show()
    plt.clf()


def ax_heatmap(ax, data):
    sns.heatmap(
        data,
        ax=ax,
        center=0,
        cmap=cmap,
        # linewidths=0.1,
        xticklabels=True,
        yticklabels=True,
    )
    
    
def ax_clustermap(data, **opt):
    """
    g.dendrogram_col.linkage
    g.dendrogram_col.dendrogram
    """
    # fig, ax = plt.subplots(figsize=figsize)
    # 
    # Create a categorical palette to identify the networks
    # part_pal = sns.husl_palette(9, s=.45)
    # part_lut = {
    #     item: part_pal[ord(item[0])-66]
    #     for item in data.columns
    # }
    # # Convert the palette to vectors that will be drawn on the side of the matrix
    # part_colors = pd.Series(data.columns, index=data.columns).map(part_lut)

    # Draw the full plot
    # method: single, complete, average, weighted, centroid, median, ward
    g = sns.clustermap(
        data,
        center=0,
        cmap=cmap,
        # method=method,
        # metric=metric,
        # row_cluster=True,
        # col_cluster=True,
        xticklabels=True,
        yticklabels=True,
        # row_colors=network_colors,
        # col_colors=part_colors,
        # dendrogram_ratio=(.1, .2),
        # cbar_pos=(.02, .32, .03, .2),
        # linewidths=.75, figsize=(12, 13)
        **opt
    )
    
    return g
    

def ax_dendrogram(ax, data=None, linkage=None, labels=None):
    labels = [
        f"{i:0>2}: {category_labels[i]}"
        for i in data.index
    ] if labels is None else labels

    linkage = hierarchy.linkage(
        data,
        metric='hamming',
        method='average',
    ) if linkage is None else linkage
    
    dendrogram = hierarchy.dendrogram(
        linkage,
        ax=ax,
        # p=30,
        # truncate_mode='lastp',
        # show_contracted=True,
        orientation='left',
        labels=labels,
        color_threshold=0,
        leaf_font_size=13,
        # color_threshold=200
    )

    ax.set_title("Hibiscus Tree", fontsize=18)
    [label.set_fontname('Microsoft Yahei') for label in ax.get_yticklabels()]
    
    ax.invert_yaxis()

    ax.set_facecolor('w')
    ax.set_xticks([])
    ax.grid(False)
    
    

