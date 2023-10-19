import typing

from torch.nn import Module, Linear, ReLU
from etnn.nn.s.chiral_node import ChiralNodeNetworkTypeS
from etnn.nn.q.chiral_node import ChiralNodeNetworkTypeQ
from etnn.nn.c.chiral_node import ChiralNodeNetworkTypeC
from etnn.nn.p.chiral_node import ChiralNodeNetworkTypeP

from etnn.nn.s.rnn import RnnNetworkTypeS
import etnn.tools.permutation_reordering as pr
from etnn.tools.tree_existance import contains_node_type
from etnn.data import TreeNode
import torch
from itertools import permutations


class LayerManagementFramework(Module):
    """
    Class that realises a layer for equivariant inputs according to a permutation definition through a permutation
    tree.
    Handles submodules of 'chiral' type that are heavily inspired by [Gainski2023]_.

    """
    def __init__(
            self,
            in_dim: int,
            tree: TreeNode,
            hidden_dim: int = 128,
            out_dim: int = 1,
            k: int = 2,
            node_type: str = "chiral",
            bidirectional: bool = False
    ):
        """
        Initializes `LayerManagementFramework`. Input to this layer is 3-dimensional:
        (set, elements_in_set, dimension_element)

        :param in_dim: Dimension of the input values - denotes the dimension of each element in the set input.
        :type in_dim: int
        :param tree: Tree determining which input to consider equal
        :type tree: TreeNode
        :param hidden_dim: Hidden dimension the elements in the sets are transferred to. Default: ``128``.
        :type hidden_dim: int
        :param out_dim: Dimension of the desired output. General dimension of output: ``(set, out_dim)``. Default: ``1``
        :type out_dim: int
        :param k: Value that determines how many elements in order to set into context with each other. Default: ``2``
        :type k: int
        :param node_type: Parameter determining which neural network modules are to be used to realize the tree
            equivariance nodes. Possible options: ``'chiral'`` for original 2-MLP connection of entries as shown in
            [Gainski2023]_ or ``'rnn'`` for a structure similar to that but using RNN's instead of MLP's to set
            elements in context to each other. Default: ``'chiral'``
        :type node_type: str
        :param bidirectional: Parameter that controls for RNN related nodes if the RNN should be bidirectional or not.
            Default: ``False``
        :type bidirectional: bool
        """
        super().__init__()
        self.embedding_layer = Linear(in_dim, hidden_dim)

        self.tree = tree
        self.tree.calc_num_elem()

        if node_type == 'chiral':
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
        elif node_type == 'rnn':
            if contains_node_type("S", tree):
                self.tree_layer_s = RnnNetworkTypeS(
                    hidden_dim=hidden_dim,
                    k=k,
                    bidirectional=bidirectional
                )
        else:
            raise NotImplementedError(f"Node type '{node_type}' not implemented. Use documentation to see available "
                                      f"options.")

        self.reduction_layers = [
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Linear(hidden_dim // 2, hidden_dim // 4),
            ReLU(),
            Linear(hidden_dim // 4, out_dim)
        ]
        return

    def forward(self, x):
        """
        Forward function as used in most pytorch modules. Returns prediction of the input data element(s). Read more
        about this in the official pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward.

        :param x: Input data to predict.
        :type x: torch.Tensor
        :return: Prediction of the module
        :rtype: torch.Tensor
        """
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
        """
        Function to act as a switch to abstract the choice of functionality for each node type. Used to call instead
        of the actual node type for better readability and reuse of functionality.

        :param embedded_x: Tensor containing the input data in an embedded form meaning of dimension ``hidden_dim``.
        :type embedded_x: torch.Tensor
        :param tree: Tree containing the tree node for which to currently act upon.
        :type tree: TreeNode
        :return: prediction/result of a subcomponent used to derive the final result (=label)
        :rtype: torch.Tensor
        """
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
            embedded_x: torch.Tensor,
            children_list: typing.List[TreeNode],
            node_module: typing.Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Function to traverse the tree nodes provided in the parameter in order to generate a prediction/label for the
        input data at this node level. Uses the provided function as an indicator of what nodetype the current node is.

        :param embedded_x: Tensor containing the input data in an embedded form meaning of dimension ``hidden_dim``.
        :type embedded_x: torch.Tensor
        :param children_list: List of nodes that are children to the current node
        :type children_list: typing.List[TreeNode]
        :param node_module: Specifies which module to use for the current node type
        :type node_module: typing.Callable[[torch.Tensor], torch.Tensor]
        :return: prediction/result of a subcomponent used to derive the final result (=label)
        :rtype: torch.Tensor
        """
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
            embedded_x: torch.Tensor,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        """
        Function realising the functionality of a S type node.

        :param embedded_x: Tensor containing the input data in an embedded form meaning of dimension ``hidden_dim``.
        :type embedded_x: torch.Tensor
        :param children_list: Defining which components are contained in the current node
        :type children_list: typing.List[TreeNode]
        :return: prediction/result of a subcomponent used to derive the final result (=label)
        :rtype: torch.Tensor
        """
        return self.ordered_tree_traversal(
            embedded_x,
            children_list,
            self.tree_layer_s
        )

    def handle_q(
            self,
            embedded_x: torch.Tensor,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        """
        Function realising the functionality of a Q type node.

        :param embedded_x: Tensor containing the input data in an embedded form meaning of dimension ``hidden_dim``.
        :type embedded_x: torch.Tensor
        :param children_list: Defining which components are contained in the current node
        :type children_list: typing.List[TreeNode]
        :return: prediction/result of a subcomponent used to derive the final result (=label)
        :rtype: torch.Tensor
        """
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
            embedded_x: torch.Tensor,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        """
        Function realising the functionality of a C type node.

        :param embedded_x: Tensor containing the input data in an embedded form meaning of dimension ``hidden_dim``.
        :type embedded_x: torch.Tensor
        :param children_list: Defining which components are contained in the current node
        :type children_list: typing.List[TreeNode]
        :return: prediction/result of a subcomponent used to derive the final result (=label)
        :rtype: torch.Tensor
        """
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
            embedded_x: torch.Tensor,
            children_list: typing.List[TreeNode]
    ) -> torch.Tensor:
        """
        Function realising the functionality of a P type node.

        :param embedded_x: Tensor containing the input data in an embedded form meaning of dimension ``hidden_dim``.
        :type embedded_x: torch.Tensor
        :param children_list: Defining which components are contained in the current node
        :type children_list: typing.List[TreeNode]
        :return: prediction/result of a subcomponent used to derive the final result (=label)
        :rtype: torch.Tensor
        """
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
