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
    """
    Class that computes regression and classification scores and saves computed data to csv file at each step so that
    no data is lost.
    """
    def __init__(
            self,
            storage_name: str,
            storage_path: str = "results",
            is_classification: bool = False,
            num_classes: int = 10,
            average: str = "micro"
    ):
        """
        Init function of the AccountManager class.

        :param storage_name: name of the file into which to store the scores
        :type storage_name: str
        :param storage_path: path to the folder where the file is to be created, default: ``results``
        :type storage_path: str
        :param is_classification: bool indicating whether classification or only regression is currently executed,
            default: ``False``.
        :type is_classification: bool
        :param num_classes: number of classes if classification is set tot true in the conditions, default: ``10``
        :type num_classes: int
        :param average: average parameter to use for the sklearn score function for parameter average as most likely
            multi-class classification will be used. Not needed for regression. Default: ``micro``.
        :type average: str
        """
        self.storage_total_path = os.path.join(storage_path, storage_name)
        self.is_classification = is_classification
        self.num_classes = num_classes
        self.class_range = np.arange(num_classes)
        self.average = average

        self.regression_metrics = {
            "mse": mean_squared_error,
            "msa": mean_absolute_error
        }

        self.classification_metrics = {
            "precision": precision_score,
            "recall": recall_score
        }

        self.mode_order = ["train", "val", "test"]

        # open file and write header
        with open(self.storage_total_path, "w") as file:
            # write header for id and epoch
            file.write("config_id, epoch")

            # write losses
            file.write(", train_loss, val_loss, test_loss")

            # write scores for train, val and test
            for mode in self.mode_order:
                # write headers for regression scores
                for regression_score_name in self.regression_metrics.keys():
                    file.write(f", {mode}_{regression_score_name}")

                # if classification, write classification scores
                if self.is_classification:
                    for classification_score_name in self.classification_metrics.keys():
                        file.write(f", {mode}_{classification_score_name}")
            file.write("\n")

    def calc_and_collect(
            self,
            config_id: str,
            epoch: int,
            train_y_true: torch.Tensor,
            train_y_pred: torch.Tensor,
            train_loss: torch.Tensor = None,
            val_y_true: torch.Tensor = None,
            val_y_pred: torch.Tensor = None,
            val_loss: torch.Tensor = None,
            test_y_true: torch.Tensor = None,
            test_y_pred: torch.Tensor = None,
            test_loss: torch.Tensor = None,
    ):
        """
        This function calculates and collects various metrics for a given configuration and epoch. It writes the results
        to a CSV file specified by the attribute storage_total_path.

        The function uses the attributes regression_metrics, classification_metrics, and mode_order to determine which
        metrics to calculate and in which order. It also uses the attribute is_classification to check if the problem is
        a classification or a regression problem.

        The function appends a new row to the CSV file with the following format:
        config_id, epoch, train_loss, val_loss, test_loss, train_regression_scores, val_regression_scores,
        test_regression_scores, train_classification_scores, val_classification_scores, test_classification_scores.

        If any of the values are missing or None, they will be written as 0. If any of the modes are not provided, they
        will be skipped and filled with 0s.

        :param config_id: The identifier of the configuration.
        :type config_id: str
        :param epoch: The number of the epoch.
        :type epoch: int
        :param train_y_true: The true labels for the training set.
        :type train_y_true: torch.Tensor
        :param train_y_pred: The predicted labels for the training set.
        :type train_y_pred: torch.Tensor
        :param train_loss: The loss value for the training set. If None, it will be written as 0.
        :type train_loss: torch.Tensor
        :param val_y_true: The true labels for the validation set.
        :type val_y_true: torch.Tensor
        :param val_y_pred: The predicted labels for the validation set.
        :type val_y_pred: torch.Tensor
        :param val_loss: The loss value for the validation set. If None, it will be written as 0.
        :type val_loss: torch.Tensor
        :param test_y_true: The true labels for the test set.
        :type test_y_true: torch.Tensor
        :param test_y_pred: The predicted labels for the test set.
        :type test_y_pred: torch.Tensor
        :param test_loss: The loss value for the test set. If None, it will be written as 0.
        :type test_loss: torch.Tensor

        :return: Nothing currently
        :rtype: None
        """
        # conver the id to string if it is not one already
        config_id = str(config_id)

        # ease latter logic
        y_true = {
            "train": train_y_true,
            "val": val_y_true,
            "test": test_y_true
        }
        y_pred = {
            "train": train_y_pred,
            "val": val_y_pred,
            "test": test_y_pred
        }

        # append to file
        with open(self.storage_total_path, "a") as file:
            # write the config id and epoch to csv file
            file.write(f"{config_id}, {epoch}")

            # write losses
            for loss in [train_loss, val_loss, test_loss]:
                if loss is not None:
                    file.write(f", {float(loss)}")
                else:
                    file.write(", 0.")

            # create scores for each mode
            for mode in self.mode_order:

                # in case the mode did not receive values
                if y_true[mode] is None or y_pred[mode] is None:
                    num_zeros = len(self.regression_metrics) + self.is_classification*len(self.classification_metrics)
                    for _ in range(num_zeros):
                        file.write(", 0.")
                    continue

                # write the regression scores
                for score in self.regression_metrics.values():
                    file.write(f", {score(y_true=y_true[mode], y_pred=y_pred[mode])}")

                # write the classification scores if the problem is a classification problem
                if self.is_classification:
                    for score in self.classification_metrics.values():
                        temp_score = score(
                            y_true=y_true[mode],
                            y_pred=y_pred[mode],
                            labels=self.class_range,
                            zero_division=0.,
                            average=self.average
                        )
                        file.write(f", {temp_score}")
            file.write("\n")
        return
