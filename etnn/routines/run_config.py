import typing

import torch
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from tqdm import tqdm

from etnn import TreeNode
from etnn.data.ferris_wheel import load_pure_ferris_wheel_dataset, load_modified_ferris_wheel_dataset
from etnn.nn.layer_framework import LayerManagementFramework
from etnn.tools.loader import create_sampler
from etnn.tools.training import train_epoch, eval_epoch
from etnn.tools.training_tools import ConfigStore, AccuracyManager, seeding_all, EpochControl
from etnn.nn.baseline import create_baseline_model, calc_params
import os
import pandas as pd


def run_config(
        config: ConfigStore,
        dataset_path: str = "./datasets",
        results_folder: str = "./results",
        check_duplicate: bool = True,
        verbose: bool = True
) -> int:
    """
    Function that runs the experiment(s) for one config. Automatically and continuously saves results.

    :param config: config to execute
    :type config: ConfigStore
    :param dataset_path: path to dataset folder.
    :type dataset_path: str
    :param results_folder: path to results folder. function will subsequently create this folder if it does not exist.
    :type results_folder: str
    :return: config id
    :rtype: int
    """
    # definition of constants
    val_perc = 0.3
    model_saving_name = "model.pt"
    config_saving_name = "config.json"
    accuracy_saving_name = "accuracies.csv"
    config_index_name = "config_index.csv"

    # DEALING WITH SAVING PATH
    config_idx, already_present = acquire_config_idx(config, config_index_name, results_folder)
    if check_duplicate and already_present:
        return config_idx

    # DEALING WITH STORAGE PATH CREATION AND CHECKS
    # if not present create the folder for this run
    storage_folder = os.path.join(results_folder, str(config_idx))
    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)

    # CHOICES FOR DATASET
    # todo: add further with more permutated elements and with invalid elements
    dataset, df_index = choice_dataset(config, dataset_path)

    # SPLITTING DATASET IN TRAIN AND VAL
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(
        dataset,
        [1 - val_perc, val_perc],
        generator=generator
    )

    # todo: rethink presence of testset in this method - yes or no. currently: no

    # ESTABLISHMENT OF LOADERS
    train_loader = choice_trainloader(config, df_index, train_ds)

    val_loader = DataLoader(val_ds, batch_size=4 * config.batch_size, shuffle=False)

    # BUILD DATASET SPECIFIC DATA STRUCTURE - FERRIS WHEEL
    tree_structure = TreeNode(
        node_type="C",
        children=[
            TreeNode("P", [TreeNode("E", config.num_part_pg)])
            for _ in range(config.num_gondolas)
        ]
    )

    # DEFINE DEVICE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SET UP ACCURACY PROTOCOLLING
    accuracy_protocoller = AccuracyManager(
        storage_name=accuracy_saving_name,
        storage_path=storage_folder,
        is_classification=False
    )

    # SEEDING
    seeding_all(config.seed)

    # DEFINE ETNN MODEL
    model = LayerManagementFramework(
        in_dim=config.in_dim,
        tree=tree_structure,
        hidden_dim=config.hidden_dim,
        out_dim=config.out_dim,
        k=config.k
    ).to(device)

    # DEFINE LOSS AND OPTIMIZER
    criterion = choice_loss(config)

    optimizer = choice_optim(config, model)

    # TRAINING OF ETNN
    epoch_control = EpochControl(
        model_save_name=model_saving_name,
        config_save_name=config_saving_name,
        saving_folder_path=storage_folder,
        tolerance=config.early_stop_tol,
        is_accuracy_score=False,
        verbose=False
    )
    # %%
    # train for N epochs
    iterating = range(config.num_max_epochs)
    if verbose:
        iterating = tqdm(iterating)

    for epoch in iterating:
        train_mean_loss, train_true_y, train_pred_y = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion
        )

        val_mean_loss, val_true_y, val_pred_y = eval_epoch(
            model,
            val_loader,
            device,
            criterion
        )

        # Not calculated for every step as the original purpose of test is to be tested for the best model/state
        # Note: this influences training and slows it down massively.
        # test_mean_loss, test_true_y, test_pred_y = eval_epoch(
        #    model,
        #    test_loader,
        #    device,
        #    criterion
        # )

        # use accuracy manager to calc accuracy metrics and save them
        accuracy_protocoller.calc_and_collect(
            config_id="etnn",
            epoch=epoch + 1,
            train_y_true=train_true_y,
            train_y_pred=train_pred_y,
            train_loss=train_mean_loss,
            val_y_true=val_true_y,
            val_y_pred=val_pred_y,
            val_loss=val_mean_loss,
            #    test_y_true=test_true_y,
            #    test_y_pred=test_pred_y,
            #    test_loss=test_mean_loss,
        )

        # check if model is better and save it
        # todo: probably not required to write config over and over again
        if epoch_control.retain_best(model, train_mean_loss, val_mean_loss, config):
            break

    # REPEAT FOR BASELINE MODEL
    seeding_all(config.seed)
    # %%
    model, _ = create_baseline_model(
        n_params=calc_params(model),
        input_dim=config.in_dim * config.num_gondolas * config.num_part_pg,
        n_layer=3,
        output_dim=1
    )
    model = model.to(device)
    # %%
    optimizer = choice_optim(config, model)
    # %%
    epoch_control = EpochControl(
        model_save_name="a",
        config_save_name="b",
        saving_folder_path=storage_folder,
        tolerance=config.early_stop_tol,
        is_accuracy_score=False
    )
    # %%
    # train for N epochs
    iterating = range(config.num_max_epochs)
    if verbose:
        iterating = tqdm(iterating)
    for epoch in iterating:
        train_mean_loss, train_true_y, train_pred_y = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion
        )

        val_mean_loss, val_true_y, val_pred_y = eval_epoch(
            model,
            val_loader,
            device,
            criterion
        )

        # use accuracy manager to calc accuracy metrics and save them
        accuracy_protocoller.calc_and_collect(
            config_id="baseline",
            epoch=epoch + 1,
            train_y_true=train_true_y,
            train_y_pred=train_pred_y,
            train_loss=train_mean_loss,
            val_y_true=val_true_y,
            val_y_pred=val_pred_y,
            val_loss=val_mean_loss,
        )

    # todo: should I include plotting (save to file)? probably will make git bloated - use accuracy plot notebook
    #   instead

    # todo: should load test set and write some values to dict and json file? or seperate final evaluation notebook?
    return config_idx


def choice_trainloader(
        config: ConfigStore,
        df_index: pd.DataFrame,
        train_ds: typing.Union[Subset, None]
) -> DataLoader:
    """
    Function realizing the choice of whether to use a weighted random sampler to even distribution issues or not
    controlled by the parameter ``use_equal_batcher``.

    :param config: config containing the parameter ``loss_name`` which contains string values that control which loss
        to use.
    :type config: ConfigStore
    :param df_index: data index framework to use for the creation of weighted random sampler
    :type df_index: pd.DataFrame
    :param train_ds: Train dataset (if subset - split) to get indices representing which entries of ``df_index`` are
        to be in the training dataloader
    :type train_ds: typing.Union[Subset, None]
    :return: training dataloader
    :rtype: DataLoader
    """
    if config.use_equal_batcher:
        sampler = create_sampler(df_index=df_index, dataset=train_ds)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    return train_loader


def choice_optim(
        config: ConfigStore,
        model: torch.nn.Module
) -> torch.optim.Optimizer:
    """
    Function realizing the choice of optimizer possible in the config parameters.

    :param config: config containing the parameter ``optimizer_name`` which contains string values that control which
        loss to use.
    :type config: ConfigStore
    :param model: Model which parameters to use for the optimizer initialization
    :type model: torch.nn.Module
    :return: torch optimizer
    :rtype: torch.optim.Optimizer
    """
    if config.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        raise Exception("wrong selection")
    return optimizer


def choice_loss(
        config: ConfigStore
):
    """
    Function realizing the choice of loss function possible in the config parameters.

    :param config: config containing the parameter ``loss_name`` which contains string values that control which loss
        to use.
    :type config: ConfigStore
    :return: torch loss function
    """
    if config.loss_name == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.loss_name == 'mae':
        criterion = torch.nn.L1Loss()
    elif config.loss_name == 'smooth-l1':
        criterion = torch.nn.SmoothL1Loss()
    else:
        raise Exception("wrong selection")
    return criterion


def choice_dataset(
        config: ConfigStore,
        dataset_path: str
) -> typing.Tuple[Dataset, pd.DataFrame]:
    """
    Function realizing the choice of dataset initialization possible in the config parameters.

    :param config: config containing the parameter ``dataset`` which contains integer values that control which dataset
        to generate.
    :type config: ConfigStore
    :param dataset_path: path to the folder where the dataset csv's are located
    :type dataset_path: str
    :return: dataset and index dataframe describing dataset
    :rtype: typing.Tuple[Dataset, pd.DataFrame]
    """
    if config.dataset == 0:
        dataset, df_index = load_pure_ferris_wheel_dataset(
            num_gondolas=config.num_gondolas,
            num_part_pg=config.num_part_pg,
            num_to_generate=config.ds_size,
            dataset_path=dataset_path,
            label_type=config.label_type,
            final_label_factor=config.final_label_factor
        )
    elif config.dataset == 1:
        dataset, df_index = load_modified_ferris_wheel_dataset(
            num_gondolas=config.num_gondolas,
            num_part_pg=config.num_part_pg,
            num_to_generate=int(config.ds_size * 0.8),
            num_valid_to_add=int(config.ds_size * 0.2),
            num_invalid_to_add=0,
            dataset_path=dataset_path,
            try_pregen=True,
            label_type=config.label_type,
            final_label_factor=config.final_label_factor
        )
    elif config.dataset == 2:
        dataset, df_index = load_modified_ferris_wheel_dataset(
            num_gondolas=config.num_gondolas,
            num_part_pg=config.num_part_pg,
            num_to_generate=int(config.ds_size * 0.6),
            num_valid_to_add=int(config.ds_size * 0.2),
            num_invalid_to_add=int(config.ds_size * 0.2),
            dataset_path=dataset_path,
            try_pregen=True,
            label_type=config.label_type,
            final_label_factor=config.final_label_factor
        )
    else:
        raise Exception("wrong selection")
    return dataset, df_index


def acquire_config_idx(
        config: ConfigStore,
        config_index_name: str,
        results_folder: str
) -> typing.Tuple[int, bool]:
    """
    Realizes acquisition of config ids which is used to determine where to store the config and the measurements/model.

    :param config: config containing all exchangeable parameters
    :type config: ConfigStore
    :param config_index_name: Name of the config index file
    :type config_index_name: str
    :param results_folder: path to the folder where the results are to be stored (not to be confused with the actual
        saving path of the metrics, model parameters and configs, this is a subfolder of this)
    :type results_folder: str
    :return: config index, bool whether this config already exists
    :rtype: typing.Tuple[int, bool]
    """
    # acquire saving path
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    # create path where the config idx file should be located
    config_idx_path = os.path.join(results_folder, config_index_name)
    # define columns of index table
    columns = ['config_idx'] + list(vars(config).keys())
    # if the config file exists - load it. else create a blank table
    if os.path.isfile(config_idx_path):
        config_table = pd.read_csv(config_idx_path)
    else:
        config_table = pd.DataFrame(columns=columns)
    # create new entry
    new_entry = pd.DataFrame(vars(config), index=[0])
    # check if config in table already
    merge = pd.merge(config_table, new_entry, on=columns[1:], how='inner')
    if len(merge) == 0:
        if len(config_table) == 0:
            config_idx = 0
        else:
            config_idx = config_table['config_idx'].max() + 1

        # add this config to idx table
        new_entry['config_idx'] = config_idx
        pd.concat([config_table, new_entry]).to_csv(config_idx_path, index=False)

        return config_idx, False
    else:
        config_idx = merge.iloc[0]['config_idx']
        return config_idx, True


def run_with_params(
        dataset_path: str = "./datasets",
        results_folder: str = "./results",
        in_dim: int = 15,
        hidden_dim: int = 128,
        out_dim: int = 1,
        k: int = 2,
        dataset: int = 0,
        ds_size: int = 10_000,
        num_gondolas: int = 10,
        num_part_pg: int = 5,
        loss_name: str = "mse",
        optimizer_name: str = 'adam',
        num_max_epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 1024,
        early_stop_tol: int = 5,
        use_equal_batcher: bool = False,
        seed: int = 420,
        label_type: str = "default",
        final_label_factor: float = 1/1000
):
    """
    Function that runs the experiment(s) for one config. Automatically and continuously saves results.

    :param dataset_path: path to dataset folder.
    :type dataset_path: str
    :param results_folder: path to results folder. function will subsequently create this folder if it does not exist.
    :type results_folder: str
    :param other_parameters: Other parameters in correspondence with config settings.
    :return: None
    """
    return run_config(
        config=ConfigStore(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            k=k,
            dataset=dataset,
            ds_size=ds_size,
            num_gondolas=num_gondolas,
            num_part_pg=num_part_pg,
            loss_name=loss_name,
            optimizer_name=optimizer_name,
            num_max_epochs=num_max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            early_stop_tol=early_stop_tol,
            use_equal_batcher=use_equal_batcher,
            seed=seed,
            label_type=label_type,
            final_label_factor=final_label_factor
        ),
        dataset_path=dataset_path,
        results_folder=results_folder
    )
