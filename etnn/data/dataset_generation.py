from etnn.data.permutation import TreeNode, generate_all_permutations
import numpy as np
from itertools import permutations
import typing


def generate_simple_multiclass_permutation(
        permutation_tree: TreeNode,
        integer_generation: bool = True,
        num_classes: int = 2
) -> typing.Tuple[np.ndarray, np.ndarray]:
    # get the number of elements
    num_elem = permutation_tree.calc_num_elem()

    # switch for case
    base_elements = None
    if integer_generation:
        base_elements = np.arange(num_elem)

    # switch for class number
    if num_classes == 2:
        label_template = np.array([-1, 1])
    elif num_classes == 3:
        label_template = np.array([-1, 0, 1])
    else:
        label_template = np.arange(num_classes)

    # element storage
    storage_x = None
    storage_y = None

    # iterate over all possible elements
    current_class_idx = 0
    for permutation in permutations(base_elements):
        # ===========================
        # for this permutation generate whole permutation group according to permutation tree
        # assign it all to one class
        # ===========================
        # transform to array
        elements = np.array(permutation)
        # check if element is already in list - hence all the other permutations would be too
        if storage_x is not None and (storage_x == elements).all(axis=1).any():
            continue
        # generate all permutations belonging to permutation tree
        element_perms = np.stack(generate_all_permutations(permutation_tree, elements))
        # include them into the storage
        if storage_x is None:
            storage_x = element_perms
        else:
            storage_x = np.concatenate([storage_x, element_perms])
        # add the labels and increment label index counter
        if storage_y is None:
            storage_y = np.tile(label_template[current_class_idx], len(element_perms))
        else:
            storage_y = np.concatenate([
                storage_y,
                np.tile(label_template[current_class_idx], len(element_perms))
            ])
        # if enough classes - terminate
        current_class_idx += 1
        if current_class_idx >= num_classes:
            break

    return storage_x, storage_y
