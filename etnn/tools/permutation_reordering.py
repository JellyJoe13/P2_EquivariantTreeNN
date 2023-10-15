from etnn.data.tree_structure import TreeNode
import typing
import numpy as np


def is_inverting_required(
        group_list: typing.List[TreeNode]
) -> bool:
    """
    Function that determines whether the order of the groups is important and hence if the nn module should also invert
    the order of the permutation nodes to cover all possible inputs.

    :param group_list: List of tree nodes belonging to one parent node
    :type group_list: typing.List[TreeNode]
    :return: Boolean indicating the need to invert.
    :rtype: bool
    """
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
) -> bool:
    """
    Function that determines whether permutations like C or P group list permutations need to be executed
    or if they are sufficiently similar so that these permutations can be skipped.

    :param group_list: List of tree nodes belonging to one parent node
    :type group_list: typing.List[TreeNode]
    :return: Boolean indicating the need to permute the inputted group list
    :rtype: bool
    """
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
