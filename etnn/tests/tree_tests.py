from etnn.data import permutation, tree_structure
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


if __name__ == "__main__":
    # run tree 3 test
    print("Running tree test 3:")
    tree3_test()
