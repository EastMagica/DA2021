#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/18 13:10
# @file     : plot.py
# @project  : ML2021
# @software : Jupyter

import matplotlib.pyplot as plt
import seaborn as sns

from eeg.basic import wave_info


# Constant
# --------

label_style = {
    'fontsize': 15,
    'fontname': 'Arial'
}

title_style = {
    'fontsize': 20,
    'fontname': 'Times New Roman'
}

legend_style = {
    # 'fontsize': 12, 
    'loc': 'upper right', 
    'shadow': True,
    'prop': {
        'size': 13,
        'weight' : 'normal',
        'family': 'Arial',
    }
}

palette = sns.color_palette('deep', n_colors=5)


# Functions
# ---------

def ax_psd(ax, psd, k=0):
    for i, v in enumerate(wave_info.keys()):
        domain, color = wave_info[v]
        x, y = psd[k, domain[0]:domain[1]:1]
        ax.plot(
            x, y, 
            label=v, color=palette[i],
            linewidth=1.8, alpha=0.75, 
            marker='.', markersize=12,
        )
        ax.set_xlabel("frequency [Hz]", **label_style)
        ax.set_ylabel("PSD [V**2/Hz]", **label_style)
        

def ax_histplot(ax, features, columns=None, color=None, log=True):
    ax_new = sns.histplot(
        ax=ax,
        data=features,
        label=columns,
        color=color,
        kde=True,
        log_scale=log
    )
    ax.set_xlabel("PSD [V**2/Hz]", **label_style)
    ax.set_ylabel("Count", **label_style)
    return ax_new

    
def ax_kdeplot(ax, features, log=True):
    ax_new = sns.kdeplot(
        ax=ax,
        data=features,
        palette=palette,
        log_scale=log
    )
    ax.set_xlabel("PSD [V**2/Hz]", **label_style)
    ax.set_ylabel("Count", **label_style)
    return ax_new


# Classes
# -------

class EEGViewerMixIn(object):
    def _create_figure(self, figsize, title):
        fig, ax = plt.subplots(figsize=figsize)
        if title is not None:
            ax.set_title(title, **title_style)
        return ax
        
    def _output_figure(self, title, show=True, save=False, form='png'):
        if save is True:
            plt.savefig(f"{self.name}_{title}.{form}", dpi=120)
        if show is True:
            plt.show()
        plt.clf()

    def show_psd(self, k=0, title='Power Spectral density', figsize=(12, 5), save=False, form='png'):
        ax = self._create_figure(figsize, title)
        ax_psd(
            ax=ax, 
            psd=self.psd,
            k=k
        )
        ax.legend(**legend_style)
        self._output_figure(title, show=True, save=save, form=form)
        
    def show_histplot_one(self, k=0, log=True, title='PSD Count', figsize=(12, 6), save=False, form='png'):
        ax = self._create_figure(figsize, title)
        features, _ = self.to_pandas()
        v = features.columns.to_list()[k]
        
        ax_histplot(
            ax=ax,
            features=features[v],
            columns=v,
            color=palette[k],
            log=log
        )

        ax.legend(**legend_style)
        self._output_figure(title, show=True, save=save, form=form)

    def show_kdeplot_all(self, log=True, title='PSD Count', figsize=(12, 6), save=False, form='png'):
        ax = self._create_figure(figsize, title)
        features, _ = self.to_pandas()
        
        ax_new = ax_kdeplot(
            ax=ax,
            features=features,
            log=log
        )
        
        ax_new.legend(
            features.columns.to_list(),
            **legend_style
        )
        self._output_figure(title, show=True, save=save, form=form)
