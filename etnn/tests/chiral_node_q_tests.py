from etnn.nn.layer_framework import LayerFramework, TreeNode
import torch
import numpy as np


def simple_test1():
    # create data
    data = torch.Tensor(np.random.rand(2, 10, 5))
    data = torch.cat([
        data[:, torch.roll(torch.arange(10), shifts=i)]
        for i in range(10)
    ])
    data = torch.cat([data, data.flip(-2)])
    print(data.shape)

    print("S node type")
    # pass through layer
    layer = LayerFramework(in_dim=5, k=2, tree=TreeNode("S", [TreeNode("E", data.shape[1])]))

    print(layer(data))

    print("Q node type")
    # pass through layer
    layer = LayerFramework(in_dim=5, k=2, tree=TreeNode("Q", [TreeNode("E", data.shape[1])]))

    print(layer(data))

    print("C node type")
    # pass through layer
    layer = LayerFramework(in_dim=5, k=2, tree=TreeNode("C", [TreeNode("E", data.shape[1])]))

    print(layer(data))

    print("P node type")
    # pass through layer
    layer = LayerFramework(in_dim=5, k=2, tree=TreeNode("P", [TreeNode("E", data.shape[1])]))

    print(layer(data))
    pass


def tree_nn_test():
    # create data
    data = torch.Tensor(np.random.rand(2, 10, 5))
    data = torch.cat([
        data[:, torch.roll(torch.arange(10), shifts=i)]
        for i in range(10)
    ])
    data = torch.cat([data, data.flip(-2)])
    print(data.shape)

    print("S node type")
    # pass through layer
    layer = LayerFramework(in_dim=5, k=2, tree=TreeNode("S", [
        TreeNode("E", 2),
        TreeNode("S", [TreeNode("E", data.shape[1]-4)]),
        TreeNode("E", 2)
    ]))

    print(layer(data))


if __name__ == "__main__":
    print("Simple test 1")
    simple_test1()

    print("Tree test1:")
    tree_nn_test()
