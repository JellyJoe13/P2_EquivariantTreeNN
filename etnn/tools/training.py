import torch
import numpy as np


def train_epoch(
        model: torch.nn.Module,
        train_loader: torch.utils.data.dataloader.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: str,
        criterion: torch.nn.Module
):
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
    def __init__(
            self,
            model_save_path: str,
            verbose: bool = True
    ):
        self.model_save_path = model_save_path
        self.current_best_eval = np.inf
        self.verbose = verbose

    def check_better(
            self,
            train_value: float,
            eval_value: float
    ):
        if self.current_best_eval > eval_value:
            self.current_best_eval = eval_value
            if self.verbose:
                print("++save++")
            return True
        else:
            return False

    def retain_best(
            self,
            model: torch.nn.Module,
            train_value: torch.Tensor,
            eval_value: torch.Tensor,
            is_accuracy_score: bool = False
    ):
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

        return
