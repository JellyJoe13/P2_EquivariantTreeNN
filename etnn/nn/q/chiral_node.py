import torch
from torch.nn import Module, Parameter
from etnn.nn.s.chiral_node import ChiralNodeNetworkTypeS


class ChiralNodeNetworkTypeQ(Module):
    """
    Class that realizes Q type node using methods described in paper_.
    ...
    References & Footnotes
    ======================
    ..paper: https://doi.org/10.1007/978-3-031-43418-1_3
    """
    def __init__(
            self,
            k: int = 2,
            hidden_dim: int = 128,
            use_state:bool = False
    ):
        """
        Function to initialize `ChiralNodeNetworkTypeQ`.
        :param k: Determines how many ordered elements to set into context with each other. Default: ``2``.
        :type k: int
        :param hidden_dim: Hidden dimension - dimension to work with in the module. Default: ``128``.
        :type hidden_dim: int
        :param use_state: Determines whether an additional Parameter should be used as bias in the module. Default:
        ``False``
        :type use_state: bool
        """
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
