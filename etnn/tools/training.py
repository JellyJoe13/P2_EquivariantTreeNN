import torch


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


