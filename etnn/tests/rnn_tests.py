import torch
import numpy as np
from etnn.nn.layer_framework import LayerManagementFramework, TreeNode


def test1():
    for bidirectional in [False, True]:
        print("==========================================")
        print(f"Bidirectional run: {str(bidirectional)}")
        # init data
        data = torch.Tensor(np.random.rand(2, 10, 5))
        data = torch.cat([data, data.flip(-2)])
        print(data.shape)

        print("S test")
        # pass through layer
        layer = LayerManagementFramework(
            in_dim=data.shape[-1],
            k=2,
            tree=TreeNode("S", [TreeNode("E", data.shape[1])]),
            node_type='rnn',
            bidirectional=bidirectional
        )

        print(layer(data))
    pass


if __name__ == "__main__":
    test1()
