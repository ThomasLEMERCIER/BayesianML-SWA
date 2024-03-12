from .swa import update_swa
import torch
import time
from src.swa_benchmark.models import Ensembling


def train_epoch(epoch, train_dl, optimizer, scheduler, criterion, model, device):
    model.train()
    train_loss = 0
    for it, (x, y) in enumerate(train_dl):
        itx = epoch * len(train_dl) + it
        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduler[itx]

        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)
    return train_loss


def test_epoch(test_dl, model, criterion, device, metric=None):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        metric_value = 0
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            if metric is not None:
                metric_value += metric(y_pred, y)
    test_loss /= len(test_dl)
    metric_value /= len(test_dl)
    return test_loss, metric_value


def batchnorm_update(loader, model, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model(x)


def train(
    train_dl,
    test_dl,
    optimizer,
    scheduler,
    criterion,
    model,
    epochs,
    device,
    metric=None,
):
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch(
            epoch, train_dl, optimizer, scheduler, criterion, model, device
        )
        test_loss, test_metric = test_epoch(test_dl, model, criterion, device, metric)

        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}{', Test metric: ' + format(test_metric, '.4f') if metric is not None else ''}, Time: {time.time() - start:.2f}s")

    return test_loss, test_metric


def swa_train(
    train_dl,
    test_dl,
    optimizer,
    scheduler,
    criterion,
    model,
    epochs,
    swa_model,
    swa_length,
    device,
    metric=None,
    return_ensemble=False,
    softmax_ensemble=True,
    bn_update=False,
):
    swa_n = 0
    if return_ensemble:
        ensemble_model = Ensembling(after_softmax=softmax_ensemble)
    else:
        ensemble_model = None

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch(
            epoch, train_dl, optimizer, scheduler, criterion, model, device
        )
        test_loss, test_metric = test_epoch(test_dl, model, criterion, device, metric)

        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}{', Test metric: ' + format(test_metric, '.4f') if metric is not None else ''}, Time: {time.time() - start:.2f}s")

        if epoch > 0 and (epoch + 1) % swa_length == 0:
            update_swa(swa_model=swa_model, model=model, n=swa_n)
            if bn_update:
                batchnorm_update(train_dl, swa_model, device)

            swa_n += 1

            swa_loss, swa_metric = test_epoch(
                test_dl, swa_model, criterion, device, metric
            )
            print(f"SWA model test loss: {swa_loss:.4f}{', SWA test metric: ' + format(swa_metric, '.4f') if metric is not None else ''}, Learning rate: {optimizer.param_groups[0]['lr'].item():.4f}")

            if return_ensemble:
                ensemble_model.add_copy(swa_model)
                ensemble_loss, ensemble_metric = test_epoch(
                    test_dl, ensemble_model, criterion, device, metric
                )

                print(f"Ensemble model test loss: {ensemble_loss:.4f}{', Ensemble test metric: ' + format(ensemble_metric, '.4f') if metric is not None else ''}")

    return test_loss, test_metric, ensemble_model

def train_epoch_graph(epoch, train_dl, optimizer, scheduler, criterion, model, device):
    model.train()
    train_loss = 0
    for it, data in enumerate(train_dl):
        itx = epoch * len(train_dl) + it
        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler[itx]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)

        optimizer.zero_grad()
        y_pred = model(x, edge_index)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)
    return train_loss

def test_epoch_graph(test_dl, model, criterion, device, metric=None):
    model.eval()
    with torch.no_grad():
        test_loss = 0
        metric_value = 0
        for data in test_dl:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)

            y_pred = model(x, edge_index)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            if metric is not None:
                metric_value += metric(y_pred, y)
    test_loss /= len(test_dl)
    metric_value /= len(test_dl)
    return test_loss, metric_value

def train_graph(train_dl, test_dl, optimizer, scheduler, criterion, model, epochs, device, metric=None):
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch_graph(epoch, train_dl, optimizer, scheduler, criterion, model, device)
        test_loss, test_metric = test_epoch_graph(test_dl, model, criterion, device, metric)

        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}{', Test metric: ' + format(test_metric, '.4f') if metric is not None else ''}, Time: {time.time() - start:.2f}s")

    return test_loss, test_metric

def swa_train_graph(train_dl, test_dl, optimizer, scheduler, criterion, model, epochs, swa_model, swa_length, device, metric=None):
    swa_n = 0
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_epoch_graph(epoch, train_dl, optimizer, scheduler, criterion, model, device)
        test_loss, test_metric = test_epoch_graph(test_dl, model, criterion, device, metric)

        print(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}{', Test metric: ' + format(test_metric, '.4f') if metric is not None else ''}, Time: {time.time() - start:.2f}s")

        if epoch > 0 and (epoch+1) % swa_length == 0:
            update_swa(swa_model=swa_model, model=model, n=swa_n)
            swa_n += 1

            swa_loss, swa_metric = test_epoch_graph(test_dl, swa_model, criterion, device, metric)
            print(f"SWA model test loss: {swa_loss:.4f}{', SWA test metric: ' + format(swa_metric, '.4f') if metric is not None else ''}, Learning rate: {optimizer.param_groups[0]['lr'].item():.4f}")

    return test_loss, test_metric
