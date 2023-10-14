from etnn.data.tree_structure import TreeNode
import typing
import numpy as np


def is_inverting_required(
        group_list: typing.List[TreeNode]
) -> bool:
    if len(group_list) == 1:
        return False

    len_array = np.array([
        child.num_elem
        for child in group_list
    ])
    type_array = np.array([
        child.node_type
        for child in group_list
    ])

    if np.all(len_array == len_array[::-1]) and np.all(type_array == type_array[::-1]):
        return False

    return True


def is_permuting_required(
        group_list: typing.List[TreeNode]
):
    if len(group_list) == 1:
        return False

    len_array = np.array([
        child.num_elem
        for child in group_list
    ])
    type_array = np.array([
        child.node_type
        for child in group_list
    ])

    same_len = np.all(len_array == len_array[0])
    same_type = np.all(type_array == type_array[0])

    if same_len and same_type:
        return False

    return True
