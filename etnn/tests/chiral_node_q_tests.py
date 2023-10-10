from etnn.nn.layer_framework import LayerFramework
import torch
import numpy as np


def simple_test1():
    # create data
    data = torch.Tensor(np.random.rand(2, 10, 5))
    data = torch.cat([data, data.flip(-2)])
    print(data)

    print("S node type")
    # pass through layer
    layer = LayerFramework(in_dim=5, k=2, temp_node_control="S")

    print(layer(data))

    print("Q node type")
    # pass through layer
    layer = LayerFramework(in_dim=5, k=2, temp_node_control="Q")

    print(layer(data))
    pass


if __name__ == "__main__":
    print("Simple test 1")
    simple_test1()
