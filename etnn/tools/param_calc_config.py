import typing

import torch

from etnn import TreeNode
from etnn.nn.baseline import create_baseline_model, calc_params
from etnn.nn.layer_framework import LayerManagementFramework
from etnn.tools.training_tools import ConfigStore


def calc_n_params_config(
        config: ConfigStore
) -> typing.Tuple[int, int]:
    """
    Function to calculate the number of parameters of a model based its configuration. The model is to be an etnn model
    or a model created with similar parameters (baseline model created by making it similar to the etnn model created
    with the function ``create_baseline_model(...)``). Will return two parameter counts - one for the etnn model and
    one for the corresponding baseline model.

    :param config: config to use for the building of the etnn and baseline model.
    :type config: ConfigStore
    :return: number of parameters in etnn and baseline model, respectively.
    :rtype: tying.Tuple[int, int]
    """
    tree_structure = TreeNode(
        node_type="C",
        children=[
            TreeNode("P", [TreeNode("E", config.num_part_pg)])
            for _ in range(config.num_gondolas)
        ]
    )

    model_tree = LayerManagementFramework(
        in_dim=config.in_dim,
        tree=tree_structure,
        hidden_dim=config.hidden_dim,
        out_dim=config.out_dim,
        k=config.k
    )

    model_baseline, _ = create_baseline_model(
        n_params=calc_params(model_tree),
        input_dim=config.in_dim * config.num_gondolas * config.num_part_pg,
        n_layer=3,
        output_dim=1
    )

    return calc_params(model_tree), calc_params(model_baseline)
