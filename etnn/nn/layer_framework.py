from torch.nn import Module, Linear, ReLU
from etnn.nn.s.chiral_node import ChiralNodeNetworkTypeS
from etnn.nn.q.chiral_node import ChiralNodeNetworkTypeQ
from etnn.nn.c.chiral_node import ChiralNodeNetworkTypeC
from etnn.nn.p.chiral_node import ChiralNodeNetworkTypeP
from etnn.data import TreeNode
import torch


class LayerFramework(Module):
    def __init__(
            self,
            in_dim: int,
            tree: TreeNode,
            hidden_dim: int = 128,
            out_dim: int = 1,
            k: int = 2
    ):
        super().__init__()
        self.embedding_layer = Linear(in_dim, hidden_dim)

        self.tree = tree

        # todo: maybe check if the layer type is needed or not to reduce model size
        self.tree_layer_s = ChiralNodeNetworkTypeS(
            hidden_dim=hidden_dim,
            k=k
        )

        self.tree_layer_q = ChiralNodeNetworkTypeQ(
            hidden_dim=hidden_dim,
            k=k
        )

        self.tree_layer_c = ChiralNodeNetworkTypeC(
            hidden_dim=hidden_dim,
            k=k
        )

        self.tree_layer_p = ChiralNodeNetworkTypeP(
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
        after_layer = self.control_hub(embedded_x, self.tree)

        # end linear layers to get from dim hidden to output
        reduction = after_layer
        for reduction_layer in self.reduction_layers:
            reduction = reduction_layer(reduction)
        return reduction

    def control_hub(
            self,
            embedded_x: torch.Tensor,
            tree: TreeNode
    ) -> torch.Tensor:
        # node type switch to handle different nodes later on more easily
        if tree.node_type == "S":
            return self.handle_s(embedded_x, tree)
        elif tree.node_type == "Q":
            return self.handle_q(embedded_x, tree)
        elif tree.node_type == "C":
            return self.handle_c(embedded_x, tree)
        elif tree.node_type == "P":
            return self.handle_p(embedded_x, tree)
        raise NotImplementedError("not implemented yet")

    def handle_s(
            self,
            embedded_x,
            tree: TreeNode
    ) -> torch.Tensor:
        return self.tree_layer_s(embedded_x)

    def handle_q(
            self,
            embedded_x,
            tree: TreeNode
    ) -> torch.Tensor:
        # todo: input permutation and layer wise calling (bottom tree up
        return self.tree_layer_q(embedded_x)

    def handle_c(
            self,
            embedded_x,
            tree: TreeNode
    ) -> torch.Tensor:
        # todo: input permutation and layer wise calling (bottom tree up
        return self.tree_layer_c(embedded_x)

    def handle_p(
            self,
            embedded_x,
            tree: TreeNode
    ) -> torch.Tensor:
        # todo: input permutation and layer wise calling (bottom tree up
        return self.tree_layer_p(embedded_x)
