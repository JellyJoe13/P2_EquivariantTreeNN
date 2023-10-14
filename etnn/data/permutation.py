import typing

from etnn.data.tree_structure import TreeNode, unroll_node
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
        inverted: bool = False,
        group_order: np.ndarray[int] = None
):
    """
    Generate the permutation for a S-type node.

    :param group_order: custom group order to traverse, default None
    :type group_order: np.ndarray[int], optional
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

    if group_order is not None:
        assert len(group_order) == len(subtree.children)

    children_elements = []

    offset = 0
    iteration_order = [subtree.children[i] for i in group_order] if group_order is not None else subtree.children
    iteration_order = iteration_order[::-1] if inverted else iteration_order
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
        element_subset: np.ndarray,
        group_order: np.ndarray[int] = None
) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
    """
    Generate the permutation for a Q-type node.

    :param subtree: tree to use for permutation generation
    :type subtree: TreeNode
    :param element_subset: set of elements to permute
    :type element_subset: np.ndarray
    :param group_order: Order of the children nodes of the node to be applied. Relevant when using this method for
    non-default applications than the Q-permutation order. Array of indexes. Default: ``None`` (meaning default order)
    :type group_order: np.ndarray[int]
    :return: permutations in list or array
    :rtype: typing.Union[typing.List[np.ndarray], np.ndarray]
    """
    perms = []

    # normal side
    perms += permutation_s(subtree, element_subset, group_order=group_order)
    # inverted side
    perms += permutation_s(subtree, element_subset, inverted=True, group_order=group_order)

    # INPUT REORDERING 1
    index = create_reverse_index(subtree)
    # normal side
    perms += permutation_s(subtree, element_subset[index], group_order=group_order)
    # inverted side
    perms += permutation_s(subtree, element_subset[index], inverted=True, group_order=group_order)

    # INPUT REORDERING 2
    index = create_reverse_index(subtree, invert=True)
    # normal side
    perms += permutation_s(subtree, element_subset[index], group_order=group_order)
    # inverted side
    perms += permutation_s(subtree, element_subset[index], inverted=True, group_order=group_order)

    # remove duplicates and return
    return fuze_permutations(perms)


def permutation_c(
        subtree: TreeNode,
        element_subset: np.ndarray
) -> typing.Union[typing.List[np.ndarray], np.ndarray]:
    """
    Generate the permutation for a C-type node.

    :param subtree: tree to use for permutation generation
    :type subtree: TreeNode
    :param element_subset: set of elements to permute
    :type element_subset: np.ndarray
    :return: permutations in list or array
    :rtype: typing.Union[typing.List[np.ndarray], np.ndarray]
    """
    perms = []
    # shiftin through possibilities and groups...
    order_idx = list(range(len(subtree.children)))
    order = subtree.children.copy()
    for _ in range(len(order)):
        # shift
        first = order[0]
        order[:-1] = order[1:]
        order[-1] = first
        first = order_idx[0]
        order_idx[:-1] = order_idx[1:]
        order_idx[-1] = first

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
        secondary_order = ranges.copy()
        for _ in range(len(ranges)):
            first = secondary_order[0]
            secondary_order[:-1] = secondary_order[1:]
            secondary_order[-1] = first
            perms += [
                permutation_q(
                    subtree,
                    element_subset[
                        list(chain(*secondary_order))
                    ],
                    group_order=order_idx
                )
            ]

    return fuze_permutations(perms)


def fuze_permutations(
        perms: typing.Union[np.ndarray, typing.List[np.ndarray]]
) -> np.ndarray:
    """
    Function that fuzes permutations of nodes with other permutations into permutations of the parent node.

    Example: [[[a,b,c]],[[d,e],[f,g]]] becomes [[a,b,c,d,e],[a,b,c,f,g]]
    :param perms: List of lists containing permutations from children permutations.
    :type perms: typing.Union[np.ndarray, typing.List[np.ndarray]]
    :return: Fuzed permutations.
    :rtype: np.ndarray
    """
    perms = np.stack(perms)
    if len(perms.shape) != 2:
        top_size = 1
        for i in perms.shape[:-1]:
            top_size *= i
        perms = perms.reshape((top_size, perms.shape[-1]))
    # remove duplicates and return
    return np.unique(perms, axis=0)


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
    for node_permutation in permutations(range(len(subtree.children))):
        ranges = []
        offset = 0
        for node_idx in node_permutation:
            node = subtree.children[node_idx]
            if node.node_type == "E":
                ranges += [
                    [offset + i]
                    for i in range(node.num_elem)
                ]
            else:
                ranges += [range(offset, offset+node.num_elem)]
            offset += node.num_elem
        for subs_permutation in permutations(ranges):
            perms += permutation_s(
                subtree,
                element_subset[
                    list(chain(*subs_permutation))
                ],
                group_order=node_permutation
            )

    # remove duplicates and return
    return fuze_permutations(perms)


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
    # unroll E nodes in tree for safety
    work_tree = unroll_node(tree)

    # fetch action to take for this node
    action = perm_function_register[work_tree.node_type]

    return action(work_tree, elements)
