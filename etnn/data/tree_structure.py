import typing
from etnn.data import DEFAULT_DATA_PATH
import json
import os


class TreeNode:
    """
    Tree class for permutation group representation
    """
    def __init__(
            self,
            node_type: str,
            children: typing.Union[list, int] = []
    ):
        """
        Creation function of class TreeNode that stores the tree in a way that can be easily saved.

        :param node_type: Type of node
        :type node_type: str
        :param children: List of children of the node or number
        of children in case it is a simple element type. Note that if the count is > 0 it emulates multiple element
        type nodes which are directly connected.
        :type children: typing.Union[list, int]
        """
        # structural elements
        self.node_type = node_type
        self.children = children

        # statistical elements/elements to make future computations easier
        self.num_elem = 0

    def to_json(self) -> dict:
        """
        Function to persist Tree and save it later on as a json

        :return: dict representing the json that is to be saved
        :rtype: dict
        """
        return {
            "node_type": self.node_type,
            "children": [
                child.to_json()
                for child in self.children
            ] if self.node_type != "E"
            else self.children
        }

    def calc_num_elem(self) -> int:
        """
        Calculate and update number of elements contained in this node

        :return: number of elements assigned to node
        :rtype: int
        """
        self.num_elem = sum(
            [
                child.calc_num_elem()
                for child in self.children
            ]
        ) if self.node_type != "E" else self.children

        return self.num_elem


def load_tree_from_json(tree: typing.Dict[str, typing.Union[str, list]]) -> TreeNode:
    """

    :param tree: tree in a json representation(aka dict)
    :type tree: typing.Dict[str, typing.Union[str, list]]
    :return: Loaded tree in basic version
    :rtype: TreeNode
    """
    tree = TreeNode(
        tree["node_type"],
        [
            load_tree_from_json(child)
            for child in tree["children"]
        ] if tree["node_type"] != "E"
        else tree["children"]
    )
    # calculate assigned element count
    tree.calc_num_elem()
    # return tree
    return tree


def save_tree(
        tree_node: TreeNode,
        file_name: str,
        folder_path: str = DEFAULT_DATA_PATH,
        pretty_save: bool = True
) -> None:
    """
    Save tree structure as a json file.

    :param tree_node: tree to be saved
    :type tree_node: TreeNode
    :param file_name: file name
    :type file_name: str
    :param folder_path: folder path (absolute or relative)
    :type folder_path: str, optional
    :param pretty_save: whether it shall be saved in a formatted way or not, default: true
    :type pretty_save: bool, optional
    """
    with open(os.path.join(folder_path, file_name), "w") as file:
        json.dump(tree_node.to_json(), file, indent=4 if pretty_save else ...)


def load_tree(
        file_name: str,
        folder_path: str = DEFAULT_DATA_PATH
) -> TreeNode:
    """
    Load tree from json file.

    :param file_name: name of the file
    :type file_name: str
    :param folder_path: folder path (relative or absolute)
    :type folder_path: str, optional
    :return: Tree constructed from json file
    :rtype: TreeNode
    """
    loaded_dict = None
    with open(os.path.join(folder_path, file_name), "r") as file:
        loaded_dict = json.load(file)

    if loaded_dict is None:
        raise Exception("File cannot be loaded")

    return load_tree_from_json(loaded_dict)


def unroll_node(
        tree: TreeNode
) -> TreeNode:
    new_children = []

    if tree.node_type == "E":
        return tree

    for child in tree.children:
        if child.node_type == "E":
            for _ in range(child.children):
                new_children += [TreeNode("E", 1)]
        else:
            new_children += [child]

    ret_tree = TreeNode(
        node_type=tree.node_type,
        children=new_children
    )
    ret_tree.calc_num_elem()
    return ret_tree
