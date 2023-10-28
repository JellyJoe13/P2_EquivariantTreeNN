import torch


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
