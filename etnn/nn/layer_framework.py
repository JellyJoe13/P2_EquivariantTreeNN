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
        self.tree.calc_num_elem()

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
        # init storage room
        data = []

        # init offset to index properly
        offset = 0

        # loop over children and add to data storage by either
        # - directly adding part of the tensor
        # - run a submodule part and reduce to one vector
        for child in tree.children:

            # if node is E then just add the part of tensor to data array
            if child.node_type == "E":
                data += [embedded_x[..., offset:offset+child.num_elem, :]]

            # else run part of the array through the corresponding nn module
            else:
                data += [
                    torch.unsqueeze(
                        self.control_hub(embedded_x[..., offset:offset+child.num_elem, :], child),
                        dim=-2
                    )
                ]

            # increase offset for indexing
            offset += child.num_elem

        # stack data and turn it into tensor to run through this node's nn module
        fuzed_data = torch.cat(data, dim=-2)

        return self.tree_layer_s(fuzed_data)

    def handle_q(
            self,
            embedded_x,
            tree: TreeNode
    ) -> torch.Tensor:
        # todo: input permutation and layer wise calling (bottom tree up
        # FIRST DIRECTION TREE NODE INTERPRETATION
        # init storage room
        data = []

        # init offset to index properly
        offset = 0

        # loop over children and add to data storage by either
        # - directly adding part of the tensor
        # - run a submodule part and reduce to one vector
        for child in tree.children:

            # if node is E then just add the part of tensor to data array
            if child.node_type == "E":
                data += [embedded_x[..., offset:offset + child.num_elem, :]]

            # else run part of the array through the corresponding nn module
            else:
                data += [
                    torch.unsqueeze(
                        self.control_hub(embedded_x[..., offset:offset + child.num_elem, :], child),
                        dim=-2
                    )
                ]

            # increase offset for indexing
            offset += child.num_elem

        # stack data and turn it into tensor to run through this node's nn module
        fuzed_data = torch.cat(data, dim=-2)

        # compute result of this node
        first = self.tree_layer_q(fuzed_data)

        # SECOND DIRECTION TREE NODE INTERPRETATION
        # todo: add checks if this second part is required or can be left out (like: are all chilren E's or all nodes
        #  of the same type and size)
        # todo: decrease overhead/make more efficient
        # init storage room
        data = []

        # init offset to index properly
        offset = 0

        # loop over children and add to data storage by either
        # - directly adding part of the tensor
        # - run a submodule part and reduce to one vector
        for child in tree.children[::-1]:

            # if node is E then just add the part of tensor to data array
            if child.node_type == "E":
                data += [embedded_x[..., offset:offset + child.num_elem, :]]

            # else run part of the array through the corresponding nn module
            else:
                data += [
                    torch.unsqueeze(
                        self.control_hub(embedded_x[..., offset:offset + child.num_elem, :], child),
                        dim=-2
                    )
                ]

            # increase offset for indexing
            offset += child.num_elem

        # stack data and turn it into tensor to run through this node's nn module
        fuzed_data = torch.cat(data, dim=-2)

        second = self.tree_layer_q(fuzed_data)

        # mean over first and second and return
        return torch.mean(
            torch.stack([first, second]),
            dim=0
        )

    def handle_c(
            self,
            embedded_x,
            tree: TreeNode
    ) -> torch.Tensor:
        # todo: input permutation and layer wise calling (bottom tree up
        # todo: efficient way of doing this?
        return self.tree_layer_c(embedded_x)

    def handle_p(
            self,
            embedded_x,
            tree: TreeNode
    ) -> torch.Tensor:
        # todo: input permutation and layer wise calling (bottom tree up
        # todo: efficient way of doing this? very inefficient otherwise or heavy logic to determine which permutations
        #  are really necessary
        return self.tree_layer_p(embedded_x)
