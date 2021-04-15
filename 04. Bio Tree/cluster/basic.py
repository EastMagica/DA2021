#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/14 14:44
# @file     : basic.py
# @project  : ML2021
# @software : Jupyter

import numpy as np


# Constant
# --------

category_labels = ['大叶木槿','黄槿','高红槿','樟叶槿','滇南芙蓉','旱地木槿','光柱旱地木槿','海滨木槿','吊灯扶桑','朱槿','重瓣朱槿','庐山芙蓉','长柄庐山芙蓉','美丽芙蓉','全叶美丽芙蓉','台湾芙蓉','木芙蓉','洋槿','贵州芙蓉','木槿','长苞木槿','短苞木槿','百花重瓣木槿','粉紫重瓣木槿','雅致木槿 ','大花木槿 ','牡丹木槿','紫花重瓣木槿','百花单瓣木槿','华木槿 ','光籽木槿 ','红秋葵 ','芙蓉葵','云南芙蓉 ','刺芙蓉','辐射刺芙蓉 ','野西瓜苗 ','玫瑰茄','大麻槿','草木槿 ','外类群蜀葵']

standarded_tree = [[[[[36, 33], [37, 39]], [[34, 35], 38]], [[[[19, 25, 21, 20, 28, 24], 26, 27], 22, 23], [[29, 30], 31]]], [[[[16, 15], 18], [12, 11]], [[[[14, 13], [6, 5]], 4], [[[[[10, 9], [32, 8]], 3], 17], [[[7, 1], 2], 0]]]], 40]

# Functions
# ---------

def get_weighted_hamming(w):
    def weighted_hamming(u, v):
        return np.sum((u != v).astype(float) * w) / u.size
    return weighted_hamming
