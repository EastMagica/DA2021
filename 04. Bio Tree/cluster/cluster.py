#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/13 12:46
# @file     : cluster.py
# @project  : ML2021
# @software : Jupyter

import abc

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

from cluster.rfdist import split_tree
from cluster.basic import get_weighted_hamming, category_labels
from cluster.plot import ax_heatmap, ax_clustermap, ax_dendrogram


# Classes
# -------

class Cluster(object):
    """
    Cluster.
    """
    def __init__(self, data, method='average', metric='hamming', imputer=None, name=''):
        self.name = name
        self.method = method
        self.metric = metric
        self.imputer = imputer

        self._data = data
        self._split_tree = None
        self._row_linkage = None
        self._col_linkage = None
        self._imputer_data = None
            
    @property
    def data(self):
        if self.imputer is None:
            return self._data
        if self._imputer_data is None:
            self._imputer_data = self.imputer.fit_transform(self._data)
        return self._imputer_data
    
    @property
    def row_linkage(self):
        if self._row_linkage is None:
            self._row_linkage = self._get_linkage(
                orientation='row'
            )
        return self._row_linkage
    
    @property
    def col_linkage(self):
        if self._col_linkage is None:
            self._col_linkage = self._get_linkage(
                orientation='col'
            )
        return self._col_linkage
    
    @row_linkage.setter
    def row_linkage(self, value):
        self._row_linkage = value
        
    @col_linkage.setter
    def col_linkage(self, value):
        self._col_linkage = value
        
    @property
    def split_tree(self):
        if self._split_tree is None:
            self._split_tree = set(split_tree(self.row_linkage))
        return self._split_tree
            
    def _get_linkage(self, orientation='row'):
        if orientation == 'row':
            data = self.data
        elif orientation == 'col':
            data = self.data.T
        return hierarchy.linkage(
            data,
            method=self.method,
            metric=self.metric,
        )
    
    def rf_distance(self, cluster2):
        if isinstance(cluster2, set):
            split_tree_2 = cluster2
        elif isinstance(cluster2, list):
            split_tree_2 = set(cluster2)
        else:
            split_tree_2 = cluster2.split_tree
        n = self.data.index.size
        n1 = len(self.split_tree)
        n2 = len(split_tree_2)
        n12 = len(self.split_tree & split_tree_2)
        return (n1 + n2 - 2 * n12) / (2 * (n - 3))


class ClusterReportMixIn(object):
    """
    Cluster Plotting MixIn Class.
    """
    def _create_figure(self, figsize, title):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return ax
        
    def _output_figure(self, title, show=True, save=False, form='png'):
        if save is True:
            plt.savefig(f"{self.name}_{title}.{form}", dpi=120)
        if show is True:
            plt.show()
        plt.clf()
        
    def show_heatmap(self, title='Heatmap', figsize=(10, 8), save=False, form='png'):
        ax = self._create_figure(figsize, title)
        ax_heatmap(
            ax=ax, 
            data=self.data
        )
        self._output_figure(title, show=True, save=save, form=form)
        
    def show_clustermap(self, title='Clustermap', figsize=(12, 12), save=False, form='png'):
        g = ax_clustermap(
            data=self.data,
            row_linkage=self.row_linkage,
            col_linkage=self.col_linkage,
        )
        if save is True:
            g.savefig(f"{self.name}_{title}.{form}", dpi=120)
    
    def show_dendrogram(self, title='Dendrogram', figsize=(7, 12), save=False, form='png'):
        ax = self._create_figure(figsize, title)
        ax_dendrogram(
            ax=ax,
            linkage=self.row_linkage,
            labels=self.labels
        )
        self._output_figure(title, show=True, save=save, form=form)


class HibiscusCluster(Cluster, ClusterReportMixIn):
    """
    
    Parameters
    ----------
    data: array_like
    linkage: array_like
    opt: dict
        method
        metric
    
    """
    def __init__(self, data, method='average', metric='hamming', imputer=None, name='', **opt):
        super().__init__(data, method, metric, imputer, name)

        self._labels = None
        
        # Weighted Hamming Distance
        if self.metric == 'hamming' and 'weight' in opt:
            self.metric = get_weighted_hamming(opt.get('weight'))

    @property
    def labels(self):
        if self._labels is None:
            self._labels = [
                f"{i:0>2}: {category_labels[i]}" for i in self.data.index
            ]
        return self._labels

    
        
    
