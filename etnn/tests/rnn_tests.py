import torch
import numpy as np
from etnn.nn.s.rnn import RnnNetworkTypeS


def test1():
    # init data
    data = torch.Tensor(np.random.rand(2, 10, 5))
    data = torch.cat([data, data.flip(-2)])
    print(data.shape)

    print("S test")
    # pass through layer
    layer = RnnNetworkTypeS(hidden_dim=5, k=2)

    print(layer(data))
    pass


if __name__ == "__main__":
    test1()
