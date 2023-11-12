import torch
from torch.utils.data import random_split, DataLoader
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
        results_folder: str = "./results"
):
    # definition of constants
    val_perc = 0.3
    model_saving_name = "model.pt"
    config_saving_name = "config.json"
    accuracy_saving_name = "accuracies.csv"
    config_index_name = "config_index.csv"

    # DEALING WITH SAVING PATH
    config_idx = acquire_config_idx(config, config_index_name, results_folder)

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
        is_accuracy_score=False
    )
    # %%
    # train for N epochs
    for epoch in tqdm(range(config.num_max_epochs)):
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
        if epoch_control.retain_best_and_stop(model, train_mean_loss, val_mean_loss, config):
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
    for epoch in tqdm(range(config.num_max_epochs)):
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

        # check if model is better and save it
        if epoch_control.should_early_stop(train_mean_loss, val_mean_loss):
            break

    # todo: should I include plotting (save to file)? probably will make git bloated - use accuracy plot notebook
    #   instead

    # todo: should load test set and write some values to dict and json file? or seperate final evaluation notebook?
    pass


def choice_trainloader(config, df_index, train_ds):
    if config.use_equal_batcher:
        sampler = create_sampler(df_index=df_index, dataset=train_ds)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    return train_loader


def choice_optim(config, model):
    if config.optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        raise Exception("wrong selection")
    return optimizer


def choice_loss(config):
    if config.loss_name == 'mse':
        criterion = torch.nn.MSELoss()
    elif config.loss_name == 'mae':
        criterion = torch.nn.L1Loss()
    elif config.loss_name == 'smooth-l1':
        criterion = torch.nn.SmoothL1Loss()
    else:
        raise Exception("wrong selection")
    return criterion


def choice_dataset(config, dataset_path):
    if config.dataset == 0:
        dataset, df_index = load_pure_ferris_wheel_dataset(
            num_gondolas=config.num_gondolas,
            num_part_pg=config.num_part_pg,
            num_to_generate=config.ds_size,
            dataset_path=dataset_path
        )
    elif config.dataset == 1:
        dataset, df_index = load_modified_ferris_wheel_dataset(
            num_gondolas=config.num_gondolas,
            num_part_pg=config.num_part_pg,
            num_to_generate=int(config.ds_size * 0.8),
            num_valid_to_add=int(config.ds_size * 0.2),
            num_invalid_to_add=0,
            dataset_path=dataset_path,
            try_pregen=True
        )
    elif config.dataset == 2:
        dataset, df_index = load_modified_ferris_wheel_dataset(
            num_gondolas=config.num_gondolas,
            num_part_pg=config.num_part_pg,
            num_to_generate=int(config.ds_size * 0.6),
            num_valid_to_add=int(config.ds_size * 0.2),
            num_invalid_to_add=int(config.ds_size * 0.2),
            dataset_path=dataset_path,
            try_pregen=True
        )
    else:
        raise Exception("wrong selection")
    return dataset, df_index


def acquire_config_idx(config, config_index_name, results_folder):
    # aquire saving path
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
    else:
        config_idx = merge.iloc[0]['config_idx']
    return config_idx


def run_with_params(
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
        seed: int = 420
):
    return run_config(
        ConfigStore(
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
            seed=seed
        )
    )
