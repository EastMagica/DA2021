#!/usr/bin/python
# -*- coding:utf-8 -*-
# @author   : east
# @time     : 2021/4/13 17:24
# @file     : rfdist.py
# @project  : ML2021
# @software : Jupyter

from scipy.cluster.hierarchy import to_tree, leaves_list


# Functions
# ---------

def get_leaves_num(Z):
    return len(leaves_list(Z))


def tree_to_dict(tree):
    root_str = f"{tree.id}"
    if tree.is_leaf():
        return root_str
    else:
        return {
            root_str: [
                tree_to_dict(tree.get_left()),
                tree_to_dict(tree.get_right()),
            ]
        }


def tree_to_str(tree):
    root_str = f"{tree.id:0>3d}"
    if tree.is_leaf():
        return root_str
    else:
        l_str = tree_to_str(tree.get_left())
        r_str = tree_to_str(tree.get_left())
        return f"{root_str}{l_str}{r_str}"


def get_leaf(tree):
    """
    get all leaves of tree.
    """
    if tree.is_leaf():
        return list()

    leaves = list()
    
    for sub_tree in [tree.get_left(), tree.get_right()]:
        if sub_tree:
            if sub_tree.is_leaf():
                leaves.append(sub_tree.id)
            else:
                leaves.extend(
                    get_leaf(sub_tree)
                )
    
    return leaves


def split_walking(tree):
    """
    get all possible split trees.
    """
    if tree.is_leaf():
        return list()

    split_sub_list = list()
    l_tree = tree.get_left()
    r_tree = tree.get_right()
    
    if l_tree.is_leaf() and r_tree.is_leaf():
        return split_sub_list
    
    if not l_tree.is_leaf():
        split_sub_list.append(get_leaf(l_tree))
        split_sub_list.extend(split_walking(l_tree))
    
    if not r_tree.is_leaf():
        split_sub_list.append(get_leaf(r_tree))
        split_sub_list.extend(split_walking(r_tree))
    
    return split_sub_list


def split_tree(Z):
    split_list = []
    
    tree = to_tree(Z)
    leaves_all = set(leaves_list(Z))
    
    split_sub_list = split_walking(tree)
    
    split_list = [
        '_'.join(
            sorted([
                ''.join([f"{item:0>2d}" for item in sorted(item)]),
                ''.join([f"{item:0>2d}" for item in sorted(list(leaves_all - set(item)))])
            ], key=len)
        )
        for item in split_sub_list
    ]
    
    return split_list


def rf_distance(cluster1, cluster2, n):
    split_tree_1 = set(cluster1) if isinstance(cluster1, (list, set)) else cluster1.split_tree
    split_tree_2 = set(cluster2) if isinstance(cluster2, (list, set)) else cluster2.split_tree

    n1 = len(split_tree_1)
    n2 = len(split_tree_2)
    n12 = len(split_tree_1 & split_tree_2)
    
    return (n1 + n2 - 2 * n12) / (2 * (n - 3))


# Functions: List Tree
# --------------------

def _get_leaf_list(tree):
    """
    get all leaves of tree.
    """
    if isinstance(tree, int):
        return list()

    leaves = list()

    for sub_tree in tree:
        if isinstance(sub_tree, int):
            leaves.append(sub_tree)
        else:
            leaves.extend(
                _get_leaf_list(sub_tree)
            )
    
    return leaves


def _split_walking_list(tree):
    """
    get all possible split trees.
    """
    if isinstance(tree, int):
        return list()

    split_sub_list = list()
    
    if all([isinstance(item, int) for item in tree]):
        return split_sub_list
    
    for item in tree:
        if isinstance(item, list):
            split_sub_list.append(_get_leaf_list(item))
            split_sub_list.extend(_split_walking_list(item))
    
    return split_sub_list


def split_list_tree(tree):
    split_list = []
    
    leaves_all = set(_get_leaf_list(tree))
    
    split_sub_list = _split_walking_list(tree)
    
    split_list = [
        '_'.join(
            sorted([
                ''.join([f"{item:0>2d}" for item in sorted(item)]),
                ''.join([f"{item:0>2d}" for item in sorted(list(leaves_all - set(item)))])
            ], key=len)
        )
        for item in split_sub_list
    ]
    
    return split_list

