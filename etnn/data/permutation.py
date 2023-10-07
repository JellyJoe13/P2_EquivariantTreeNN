import typing

from etnn.data.tree_structure import TreeNode
import numpy as np
from itertools import product, chain, permutations


def permutation_e(
        subtree: TreeNode,
        element_subset: np.ndarray
) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
    """
    Generate the permutation for an E-type node.

    :param subtree: tree to use for permutation generation
    :type subtree: TreeNode
    :param element_subset: set of elements to permute
    :type element_subset: np.ndarray
    :return: permutations in list or array
    :rtype: typing.Union[typing.List[np.ndarray], np.ndarray]
    """
    assert subtree.num_elem == len(element_subset)
    return [element_subset]


def permutation_s(
        subtree: TreeNode,
        element_subset: np.ndarray,
        inverted: bool = False
):
    """
    Generate the permutation for a S-type node.

    :param subtree: tree to use for permutation generation
    :type subtree: TreeNode
    :param element_subset: set of elements to permute
    :type element_subset: np.ndarray
    :param inverted: Tells the function to invert the tree (but not the elements), default false
    :type inverted: bool, optional
    :return: permutations in list or array
    :rtype: typing.Union[typing.List[np.ndarray], np.ndarray]
    """
    assert subtree.num_elem == len(element_subset)
    children_elements = []

    offset = 0
    iteration_order = subtree.children[::-1] if inverted else subtree.children
    for child in iteration_order:
        children_elements += [
            generate_all_permutations(child, element_subset[offset:(offset+child.num_elem)])
        ]
        offset += child.num_elem

    perms = []
    for child_perms in children_elements:
        if len(perms) == 0:
            perms = child_perms
            continue
        else:
            perms = [
                np.concatenate([a, b])
                for a, b in product(perms, child_perms)
            ]

    return perms


def permutation_q(
        subtree: TreeNode,
        element_subset: np.ndarray
):
    """
    Generate the permutation for a Q-type node.

    :param subtree: tree to use for permutation generation
    :type subtree: TreeNode
    :param element_subset: set of elements to permute
    :type element_subset: np.ndarray
    :return: permutations in list or array
    :rtype: typing.Union[typing.List[np.ndarray], np.ndarray]
    """
    perms = []

    # normal side
    perms += permutation_s(subtree, element_subset)
    # inverted side
    perms += permutation_s(subtree, element_subset, inverted=True)

    # INPUT REORDERING 1
    index = create_reverse_index(subtree)
    # normal side
    perms += permutation_s(subtree, element_subset[index])
    # inverted side
    perms += permutation_s(subtree, element_subset[index], inverted=True)

    # INPUT REORDERING 2
    index = create_reverse_index(subtree, invert=True)
    # normal side
    perms += permutation_s(subtree, element_subset[index])
    # inverted side
    perms += permutation_s(subtree, element_subset[index], inverted=True)

    # remove duplicates and return
    return np.unique(
        np.stack(perms),
        axis=0
    )


def permutation_c(
        subtree: TreeNode,
        element_subset: np.ndarray
):
    perms = []
    # shiftin through possibilities and groups...
    # todo
    order = subtree.children.copy()
    for _ in range(len(order)):
        # shift
        first = order[0]
        order[:-1] = order[1:]
        order[-1] = first

        # do for all the permutations of elements
        ranges = []
        offset = 0
        for node in order:
            if node.node_type == "E":
                ranges += [
                    [offset + i]
                    for i in range(node.num_elem)
                ]
            else:
                ranges += [range(offset, offset + node.num_elem)]
            offset += node.num_elem
        for subs_permutation in permutations(ranges):
            perms += [
                permutation_q(subtree, element_subset[
                    list(chain(*subs_permutation))
                ])
            ]


    # remove duplicates and return
    return np.unique(
        np.stack(perms),
        axis=0
    )


def permutation_p(
        subtree: TreeNode,
        element_subset: np.ndarray
):
    """
    Generate the permutation for a P-type node.

    :param subtree: tree to use for permutation generation
    :type subtree: TreeNode
    :param element_subset: set of elements to permute
    :type element_subset: np.ndarray
    :return: permutations in list or array
    :rtype: typing.Union[typing.List[np.ndarray], np.ndarray]
    """
    perms = []
    for node_permutation in permutations(subtree.children):
        ranges = []
        offset = 0
        for node in node_permutation:
            if node.node_type == "E":
                ranges += [
                    [offset + i]
                    for i in range(node.num_elem)
                ]
            else:
                ranges += [range(offset, offset+node.num_elem)]
            offset += node.num_elem
        for subs_permutation in permutations(ranges):
            perms += [
                permutation_s(subtree, element_subset[
                    list(chain(*subs_permutation))
                ])
            ]

    # remove duplicates and return
    return np.unique(
        np.stack(perms),
        axis=0
    )


def create_reverse_index(
        subtree: TreeNode,
        invert: bool = False
) -> list:
    """
    Create an reverse element index based of the tree. Can be inverted.

    :param subtree: Tree to use for the group definitions
    :type subtree: TreeNode
    :param invert: Whether to invert the tree permutation group order or not, default false
    :type invert: bool, optional
    :return: index to reorder elements
    :rtype: list
    """
    index = []
    offset = 0
    iteration_order = subtree.children[::-1] if invert else subtree.children
    for child in iteration_order:
        if child.node_type == "E":
            index += [
                [offset + i]
                for i in range(child.num_elem)
            ]
        else:
            index += [
                range(offset, offset + child.num_elem)
            ]
        offset += child.num_elem
    index = index[::-1]
    index = list(chain(*index))
    return index


perm_function_register = {
    "E": permutation_e,
    "S": permutation_s,
    "Q": permutation_q,
    "P": permutation_p,
    "C": permutation_c
}


def generate_all_permutations(
        tree: TreeNode,
        elements: np.ndarray
) -> list:  # list of numpy arrays or larger dim numpy array (better latter option)
    """
    Generate the permutation for a permutation tree.

    :param tree: tree to use for permutation generation
    :type tree: TreeNode
    :param elements: set of elements to permute
    :type elements: np.ndarray
    :return: permutations in list or array
    :rtype: typing.Union[typing.List[np.ndarray], np.ndarray]
    """
    # fetch action to take for this node
    action = perm_function_register[tree.node_type]

    return action(tree, elements)
