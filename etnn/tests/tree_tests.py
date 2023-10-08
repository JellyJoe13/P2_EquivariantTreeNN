from etnn.data import permutation, tree_structure
from etnn.data.tree_structure import TreeNode
import numpy as np


def tree3_test():
    print("Test with Q root")
    # load tree
    tree = tree_structure.load_tree("tree3.json", folder_path="../../datasets")

    # part 1
    print("With S node in middle")
    # generate permutations
    perms = permutation.generate_all_permutations(tree, np.arange(tree.num_elem))

    print(perms)

    # part 2
    print("With Q node in middle")
    tree.children[1].node_type = "Q"

    # generate permutations
    perms = permutation.generate_all_permutations(tree, np.arange(tree.num_elem))

    print(perms)
    return


def tree4_test():
    tree = tree_structure.load_tree("tree4.json", folder_path="../../datasets")

    print("P as root node")
    # generate permutations
    perms = permutation.generate_all_permutations(tree, np.array([
        chr(i+97) for i in range(tree.num_elem)
    ]))

    print(perms)

    print("C as root node")
    tree.node_type = "C"
    # generate permutations
    perms = permutation.generate_all_permutations(tree, np.array([
        chr(i + 97) for i in range(tree.num_elem)
    ]))

    print(perms)


def tree5_test():
    tree = TreeNode("C", [TreeNode("E", 3)])
    tree.calc_num_elem()
    perms = permutation.generate_all_permutations(tree, np.arange(tree.num_elem))
    print(perms)
    return


if __name__ == "__main__":
    # run tree 3 test
    print("Running tree test 3:")
    tree3_test()
    print("Running tree test 4:")
    tree4_test()
    print("Running tree test 5:")
    tree5_test()
