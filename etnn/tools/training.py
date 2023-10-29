import torch
import numpy as np


def train_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.dataloader.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        criterion: torch.nn.Module
) -> torch.Tensor:
    """
    Function that trains the model for one epoch using the provided loader and collects loss data which is then
    returned.

    :param model: model to train for one epoch
    :type model: torch.nn.Module
    :param train_loader: dataloader for the trainset
    :type train_loader: torch.utils.data.dataloader.DataLoader
    :param optimizer: optimizer to use for training
    :type optimizer: torch.optim.Optimizer
    :param device: device to train on
    :type device: str
    :param criterion: criterion to calculate the loss
    :type criterion: torch.nn.Module
    :return: tensor containing the averaged loss over the batches
    :rtype: torch.Tensor
    """
    # set model to training mode
    model.train()

    # loss storage
    loss_storage = []

    for batch_data, batch_label in train_loader:
        # optimizer zero grad
        optimizer.zero_grad()

        # put data to device
        batch_data = batch_data.to(device)
        batch_label = batch_label.to(device)

        # put through model
        prediction = model(batch_data).flatten()

        # calculate loss
        loss = criterion(prediction, batch_label)

        # backward loss
        loss.backward()

        # optimizer step
        optimizer.step()

        # save loss
        loss_storage += [loss.detach().cpu()]

    return torch.mean(
        torch.stack(loss_storage)
    )


def eval_epoch(
        model: torch.nn.Module,
        eval_loader: torch.utils.data.dataloader.DataLoader,
        device: str,
        criterion: torch.nn.Module
):
    """
    Function that evaluates the model for one epoch using the provided loader and collects loss data which is then
    returned.

    :param model: model to train for one epoch
    :type model: torch.nn.Module
    :param eval_loader: dataloader for the evaluation set
    :type eval_loader: torch.utils.data.dataloader.DataLoader
    :param device: device to train on
    :type device: str
    :param criterion: criterion to calculate the loss
    :type criterion: torch.nn.Module
    :return: tensor containing the averaged loss over the batches
    :rtype: torch.Tensor
    """
    with torch.no_grad():
        # init loss storage
        loss_storage = []

        # set model to evaluation mode
        model.eval()

        for batch_data, batch_label in eval_loader:
            # put data to device
            batch_data = batch_data.to(device)
            batch_label = batch_label.to(device)

            # put through model
            prediction = model(batch_data).flatten()

            # calculate loss
            loss = criterion(prediction, batch_label)

            # append loss
            loss_storage += [loss.detach().cpu()]

        # return averaged loss
        return torch.mean(
            torch.stack(loss_storage)
        )


class EpochControl:
    """
    Tool for saving model states and later on maybe also for tracking loss data and early stopping control.
    """
    def __init__(
            self,
            model_save_path: str,
            verbose: bool = True
    ):
        """
        Init function of the EpochControl object

        :param model_save_path: specifies where the model parameters shall be saved
        :type model_save_path: str
        :param verbose: determines whether prints to console concerning the progress shall be printed or not, default:
            ``True``
        :type verbose: bool
        """
        self.model_save_path = model_save_path
        self.current_best_eval = np.inf
        self.verbose = verbose

    def check_better(
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
        if self.current_best_eval > eval_value:
            self.current_best_eval = eval_value
            if self.verbose:
                print("++save++")
            return True
        else:
            return False

    def retain_best_and_stop(
            self,
            model: torch.nn.Module,
            train_value: torch.Tensor,
            eval_value: torch.Tensor,
            is_accuracy_score: bool = False
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
        :param is_accuracy_score: Provides information whether the provided score is an accuracy score or not, default:
            ``False``
        :return: Truth value if training should be stopped or not.
        :rtype: bool
        """
        # copy values
        working_train = float(train_value)
        working_eval = float(eval_value)

        # if accuracy values - invert scores
        if is_accuracy_score:
            working_train *= (-1)
            working_eval *= (-1)

        # check if value is better, save model
        if self.check_better(train_value, eval_value):
            torch.save(model.state_dict(), self.model_save_path)

        # todo
        return False
