import torch
from torch.nn import Module, Parameter
from etnn.nn.s.chiral_node import ChiralNodeNetworkTypeS


class ChiralNodeNetworkTypeQ(Module):
    def __init__(
            self,
            k: int = 2,
            hidden_dim: int = 128,
            use_state:bool = False
    ):
        super().__init__()
        self.k = k
        self.s_module = ChiralNodeNetworkTypeS(
            k=k,
            hidden_dim=hidden_dim,
            use_state=use_state
        )
        self.use_state = use_state
        if self.use_state:
            self.own_params = Parameter(torch.empty((1, hidden_dim)))
        return

    def forward(self, x):
        # run through forward
        res_forward = self.s_module(x)
        # run through backwards
        res_backward = self.s_module(x.flip(-2))
        # stack both and aggregate with mean to preserve meaning of sum coupled with the number of elements of a node
        aggregated = torch.mean(torch.stack([res_forward, res_backward]), dim=0)
        # add bias (parameters)
        if self.use_state:
            aggregated += self.own_params
        # return
        return aggregated
