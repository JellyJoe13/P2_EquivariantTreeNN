from torch.nn import Module, Linear, ReLU
from etnn.nn.s.chiral_node import ChiralNodeNetworkTypeS
from etnn.nn.q.chiral_node import ChiralNodeNetworkTypeQ
from etnn.nn.c.chiral_node import ChiralNodeNetworkTypeC


class LayerFramework(Module):
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int = 128,
            out_dim: int = 1,
            k: int = 2,
            temp_node_control: str = "S"
    ):
        super().__init__()
        self.embedding_layer = Linear(in_dim, hidden_dim)

        if temp_node_control == "S":
            self.tree_layer = ChiralNodeNetworkTypeS(
                hidden_dim=hidden_dim,
                k=k
            )
        elif temp_node_control == "Q":
            self.tree_layer = ChiralNodeNetworkTypeQ(
                hidden_dim=hidden_dim,
                k=k
            )
        elif temp_node_control == "C":
            self.tree_layer = ChiralNodeNetworkTypeC(
                hidden_dim=hidden_dim,
                k=k
            )

        self.reduction_layers = [
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, hidden_dim // 4),
            ReLU(),
            Linear(hidden_dim // 4, out_dim)
        ]
        return

    def forward(self, x):
        # embed the input through a linear layer
        embedded_x = self.embedding_layer(x)

        # run through tree layer
        after_layer = self.tree_layer(embedded_x)

        # end linear layers to get from dim hidden to output
        reduction = after_layer
        for reduction_layer in self.reduction_layers:
            reduction = reduction_layer(reduction)
        return reduction
