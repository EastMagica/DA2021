#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/13 15:26
# @file     : pipline.py
# @project  : ML2021
# @software : Jupyter

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin


class MissingCategoryImputer(TransformerMixin):
    """
    分类型缺失值填充器
    """
    def __init__(self, cols=None):
        self.cols = cols
    
    def transform(self, df):
        x = df.copy()
        for col in self.cols:
            # 均值填充
#             x.loc[x[col] == -1, col] = int(np.around(df[col].mean()))
            # 众数填充
            x.loc[x[col] == -1, col] = x[col].value_counts().index[0]
            # 创建新类型
            # x.loc[x[col] == -2, col] = int(x[col].value_counts().index.max()+1)
            # 随机填充
            # x.loc[x[col] == -1, col] = np.random.choice(x[col].value_counts().index)
            # x[col].fillna(
            #     # 常见类别填充
            #     # x[col].value_counts().index[0],
            #     # 均值填充
            #     int(np.around(df['B2'].mean())),
            #     downcast='int16',
            #     inplace=True
            # )
        return x
    
    def fit(self, *_):
        return self
    

class InapplicableCategoryImputer(TransformerMixin):
    """
    分类型非适用值填充器
    """
    def __init__(self, cols=None):
        self.cols = cols
    
    def transform(self, df):
        x = df.copy()
        for col in self.cols:
            # 创建新类型
            x.loc[x[col] == -2, col] = int(x[col].value_counts().index.max()+1)
        return x
    
    def fit(self, *_):
        return self
    
    