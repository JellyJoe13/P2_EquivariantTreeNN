import typing

from torch.nn import Module, Linear, ReLU
from etnn.nn.s.chiral_node import ChiralNodeNetworkTypeS
from etnn.nn.q.chiral_node import ChiralNodeNetworkTypeQ
from etnn.nn.c.chiral_node import ChiralNodeNetworkTypeC
from etnn.nn.p.chiral_node import ChiralNodeNetworkTypeP
import etnn.tools.permutation_reordering as pr
from etnn.tools.tree_existance import contains_node_type
from etnn.data import TreeNode
import torch
from itertools import permutations


class ChiralLayerManagementFramework(Module):
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

        if contains_node_type("S", tree):
            self.tree_layer_s = ChiralNodeNetworkTypeS(
                hidden_dim=hidden_dim,
                k=k
            )

        if contains_node_type("Q", tree):
            self.tree_layer_q = ChiralNodeNetworkTypeQ(
                hidden_dim=hidden_dim,
                k=k
            )

        if contains_node_type("C", tree):
            self.tree_layer_c = ChiralNodeNetworkTypeC(
                hidden_dim=hidden_dim,
                k=k
            )

        if contains_node_type("P", tree):
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
        # todo(potential): get rid of recursive calling with dynamic routines
        # todo(potential): increase efficiency
        # node type switch to handle different nodes later on more easily
        if tree.node_type == "S":
            return self.handle_s(embedded_x, tree.children)
        elif tree.node_type == "Q":
            return self.handle_q(embedded_x, tree.children)
        elif tree.node_type == "C":
            return self.handle_c(embedded_x, tree.children)
        elif tree.node_type == "P":
            return self.handle_p(embedded_x, tree.children)
        raise NotImplementedError("not implemented yet")

    def ordered_tree_traversal(
            self,
            embedded_x,
            children_list: typing.List[TreeNode],
            node_module
    ):
        # init storage room
        data = []

        # init offset to index properly
        offset = 0

        # loop over children and add to data storage by either
        # - directly adding part of the tensor
        # - run a submodule part and reduce to one vector
        for child in children_list:

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

        return node_module(fuzed_data)

    def handle_s(
            self,
            embedded_x,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        return self.ordered_tree_traversal(
            embedded_x,
            children_list,
            self.tree_layer_s
        )

    def handle_q(
            self,
            embedded_x,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        # FIRST DIRECTION TREE NODE INTERPRETATION
        first = self.ordered_tree_traversal(
            embedded_x,
            children_list,
            self.tree_layer_q
        )

        # SECOND DIRECTION TREE NODE INTERPRETATION
        # check if inverting action is required
        if not pr.is_inverting_required(children_list):
            return first
        second = self.ordered_tree_traversal(
            embedded_x,
            children_list[::-1],
            self.tree_layer_q
        )

        # mean over first and second and return
        return torch.mean(
            torch.stack([first, second]),
            dim=0
        )

    def handle_c(
            self,
            embedded_x,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        # init variables
        n_c = len(children_list)

        # copy children list
        group_list = children_list.copy()

        # init results storage
        results_storage = []

        # check if cycling is required in the first place to avoid overhead
        permutation_needed = pr.is_permuting_required(group_list)

        # shift over input group wise and also invert
        for _ in range(n_c):
            # do q layer for arrangement basically
            # FIRST
            results_storage += [
                self.ordered_tree_traversal(
                    embedded_x=embedded_x,
                    children_list=group_list,
                    node_module=self.tree_layer_c
                )
            ]
            # SECOND
            if pr.is_inverting_required(children_list):
                results_storage += [
                    self.ordered_tree_traversal(
                        embedded_x=embedded_x,
                        children_list=group_list[::-1],
                        node_module=self.tree_layer_c
                    )
                ]

            if not permutation_needed:
                break

            # shift one position
            first = group_list[0]
            group_list[:-1] = group_list[1:]
            group_list[-1] = first

        # mean over first and second and return
        return torch.mean(
            torch.stack(results_storage),
            dim=0
        )

    def handle_p(
            self,
            embedded_x,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        # init data storage
        data_storage = []

        # evaluate if permuting in the first place is required or not
        permuting_needed = pr.is_permuting_required(children_list)

        # generate all permutations of this node
        for node_perm in permutations(children_list):
            data_storage += [
                self.ordered_tree_traversal(
                    embedded_x=embedded_x,
                    children_list=node_perm,
                    node_module=self.tree_layer_p
                )
            ]

            if not permuting_needed:
                break

        return torch.mean(
            torch.stack(data_storage),
            dim=0
        )
