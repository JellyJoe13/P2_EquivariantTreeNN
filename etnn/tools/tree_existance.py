from etnn.data.tree_structure import TreeNode


def contains_node_type(
        node_type: str,
        tree: TreeNode
) -> bool:
    # if this is the node we are looking for then return True
    if node_type == tree.node_type:
        return True

    # else search in children layer
    # hence check if it has children or not
    elif tree.node_type == "E":
        return False

    # there are children, search in them
    else:
        # recurse for all children
        for child in tree.children:
            if contains_node_type(node_type, child):
                return True

        return False
