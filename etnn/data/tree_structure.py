import typing


class TreeNode:
    """
    Tree class for basic operations and storage - there will be a more advanced class which pre-computes relevant
    elements beforehand like e.g. the number of elements assigned to a node or allowed permutations
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
        self.node_type = node_type
        self.children = children

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


def load_tree_from_json(tree: typing.Dict[str, typing.Union[str, list]]) -> TreeNode:
    """

    :param tree: tree in a json representation(aka dict)
    :type tree: typing.Dict[str, typing.Union[str, list]]
    :return: Loaded tree in basic version
    :rtype: TreeNode
    """
    return TreeNode(
        tree["node_type"],
        [
            load_tree_from_json(child)
            for child in tree["children"]
        ] if tree["node_type"] != "E"
        else tree["children"]
    )
