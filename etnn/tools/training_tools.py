import json
import os

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score


class ConfigStore:
    """
    Class for storing configuration which can be saved and loaded to/from a json file
    """
    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            out_dim: int,
            k: int,
            is_accuracy_score: bool,
            which_score: str,
            loss_name: str,
            optimizer_name: str,
            num_max_epochs: int = 20,
            learning_rate: float = 0.001,
            batch_size: int = 32,
            early_stop_tol: int = 5,
            is_classification: bool = False,
    ):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.k = k

        self.is_accuracy_score = is_accuracy_score
        self.is_classification = is_classification
        self.which_score = which_score

        self.loss_name = loss_name
        self.optimizer_name = optimizer_name

        self.num_max_epochs = num_max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.early_stop_tol = early_stop_tol


def config_to_json(
        config: ConfigStore,
        saving_path: str
):
    with open(saving_path, 'w') as file:
        json.dump(vars(config), file, indent=4)
    return


def load_config(
        file_path: str
) -> ConfigStore:
    with open(file_path, 'r') as file:
        data = json.load(file)
        return ConfigStore(**data)
    return None


class EpochControl:
    """
    Tool for saving model states and later on maybe also for tracking loss data and early stopping control.
    """
    def __init__(
            self,
            model_save_name: str = "model.pt",
            config_save_name: str = "config.json",
            saving_folder_path: str = ".",
            tolerance: int = 5,
            verbose: bool = True,
            is_accuracy_score: bool = False
    ):
        """
        Init function of the EpochControl object

        :param model_save_name: specifies under which name to save the model
        :type model_save_name: str
        :param config_save_name: specifies under which name to save the config, default: ``config.json``
        :type config_save_name: str
        :param saving_folder_path: folder in which to save config and model files, default: ``.``
        :type saving_folder_path: str
        :param verbose: determines whether prints to console concerning the progress shall be printed or not, default:
            ``True``
        :type verbose: bool
        :param is_accuracy_score: Provides information whether the provided score is an accuracy score or not, default:
            ``False``
        :type is_accuracy_score: bool
        """
        self.model_save_path = os.path.join(saving_folder_path, model_save_name)
        self.config_save_path = os.path.join(saving_folder_path, config_save_name)
        self.current_best_save = np.inf
        self.current_best_stop = np.inf
        self.verbose = verbose
        self.num_epochs_not_better = 0
        self.tolerance = tolerance
        self.is_accuracy_score = is_accuracy_score

    def check_better_save(
            self,
            train_value: float,
            eval_value: float
    ) -> bool:
        """
        Function that implements logic to determine whether this model performance is considered better or not.

        :param train_value: training score of this epoch
        :type train_value: float
        :param eval_value: evaluation score of this epoch
        :type eval_value: float
        :return: Whether this epoch's state is considered better
        :rtype: bool
        """
        if self.current_best_save > eval_value:
            self.current_best_save = eval_value
            return True
        else:
            return False

    def should_early_stop(
            self,
            train_value: float,
            eval_value: float
    ) -> bool:
        """
        Function determining whether the training should stop or not.

        :param train_value: value coming from training
        :type train_value: float
        :param eval_value: value coming from validation/testing
        :type eval_value: float
        :return: whether to stop training or not
        :rtype: bool
        """
        if self.current_best_stop > eval_value:
            self.current_best_stop = eval_value
            self.num_epochs_not_better = 0
            return False
        else:
            self.num_epochs_not_better += 1
            return self.num_epochs_not_better > self.tolerance

    def retain_best_and_stop(
            self,
            model: torch.nn.Module,
            train_value: torch.Tensor,
            eval_value: torch.Tensor,
            config: ConfigStore = None
    ) -> None:
        """
        Determines based on the provided train and eval values if the current state of the model is better and shall
        be saved. Returns a truth value if training should be stopped or not.

        :param model: model which parameters to be saved
        :type model: torch.nn.Module
        :param train_value: value score produced by the train set
        :type train_value: torch.Tensor
        :param eval_value: value score produced by either the valuation or test set
        :type train_value: torch.Tensor
        :param config: Config container that will be saved if model is saved to know which model parameters to use
            when restoring model/setup, default: ``None`` meaning nothing will be saved, although not recommended.
        :type config: ConfigStore
        :return: Truth value if training should be stopped or not.
        :rtype: bool
        """
        # copy values
        working_train = float(train_value)
        working_eval = float(eval_value)

        # if accuracy values - invert scores
        if self.is_accuracy_score:
            working_train *= (-1)
            working_eval *= (-1)

        # check if value is better, save model
        if self.check_better_save(train_value, eval_value):
            if self.verbose:
                print("++save++")
            torch.save(model.state_dict(), self.model_save_path)
            if config is not None:
                config_to_json(config, self.config_save_path)

        # return truth value if to stop or not
        return self.should_early_stop(working_train, working_eval)


class AccuracyManager:
    def __init__(
            self,
            storage_name: str,
            storage_path: str = "results",
            is_classification: bool = False
    ):
        self.storage_total_path = os.path.join(storage_path, storage_name)
        self.is_classification = is_classification

        self.regression_metrics = {
            "mse": mean_squared_error,
            "msa": mean_absolute_error
        }

        self.classification_metrics = {
            "precision": precision_score,
            "recall": recall_score
        }

        # open file and write header
        with open(self.storage_total_path, "w") as file:
            file.write("config_id, epoch")
            for regression_score_name in self.regression_metrics.keys():
                file.write(f", {regression_score_name}")
            # if classification
            if self.is_classification:
                for classification_score_name in self.classification_metrics.keys():
                    file.write(f", {classification_score_name}")
            file.write("\n")

    def calc_and_collect(
            self,
            y_true: torch.Tensor,
            y_pred: torch.Tensor,
            config_id: str,
            epoch: int
    ):
        config_id = str(config_id)

        with open(self.storage_total_path, "a") as file:
            file.write(f"{config_id}, {epoch}")
            for score in self.regression_metrics.values():
                file.write(f", {score(y_true=y_true, y_pred=y_pred)}")
            if self.is_classification:
                for score in self.classification_metrics.values():
                    file.write(f", {score(y_true=y_true, y_pred=y_pred)}")
            file.write("\n")
        pass
