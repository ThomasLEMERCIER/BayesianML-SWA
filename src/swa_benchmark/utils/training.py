from .swa import update_swa
import torch
import time
from src.swa_benchmark.models import Ensembling


def train(
    model, train_dl, test_dl, criterion, optimizer, epochs, scheduler, metric=None
):
    device = next(model.parameters()).device

    test_loss = 0
    for e in range(epochs):
        model.train()
        start = time.time()
        train_loss = 0
        for it, (x, y) in enumerate(train_dl):
            itx = it + e * len(train_dl)
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
        if metric is not None:
            test_loss, test_metric = eval(model, test_dl, criterion, metric)
        else:
            test_loss = eval(model, test_dl, criterion)

        print(
            f"Epoch {e+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Time: {time.time() - start:.2f} s"
        )

        if metric is not None:
            print("Test metric: ", test_metric)

    return test_loss


def swa_train(
    model,
    swa_model,
    train_dl,
    test_dl,
    criterion,
    optimizer,
    epochs,
    swa_length,
    swa_start,
    scheduler,
    return_ensemble=False,
    metric=None,
):
    swa_n = 0
    device = next(model.parameters()).device
    if return_ensemble:
        ensemble = Ensembling()

    test_loss = 0
    for e in range(epochs):
        model.train()
        start = time.time()
        train_loss = 0
        for it, (x, y) in enumerate(train_dl):
            itx = it + e * len(train_dl)
            for param_group in optimizer.param_groups:
                param_group["lr"] = scheduler[itx]

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if e > swa_start and (e + 1) % swa_length == 0:
            update_swa(swa_model, model, swa_n)
            swa_n += 1
            swa_loss = eval(swa_model, test_dl, criterion)
            print(f"SWA test loss: {swa_loss:.4f}")

            if return_ensemble:
                ensemble.add_copy(model)
                if metric is not None:
                    ensemble_test_loss, ensemble_test_metric = eval(
                        ensemble, test_dl, criterion, metric
                    )
                    print(
                        f"Ensemble test loss: {ensemble_test_loss:.4f}, Ensemble test metric: {ensemble_test_metric:.4f}"
                    )
                else:
                    ensemble_test_loss = eval(ensemble, test_dl, criterion)
                    print(f"Ensemble test loss: {ensemble_test_loss:.4f}")

        train_loss /= len(train_dl)
        if metric is not None:
            test_loss, test_metric = eval(model, test_dl, criterion, metric)
        else:
            test_loss = eval(model, test_dl, criterion, metric)

        print(
            f"Epoch {e+1}/{epochs}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Time: {time.time() - start:.2f} s, {' (SWA update) with learning rate at: ' + str(optimizer.param_groups[0]['lr'].item()) if e > swa_start and (e+1) % swa_length == 0 else ''}"
        )

        if metric is not None:
            print("Test metric: ", test_metric)

    swa_model.train()
    with torch.no_grad():
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            _ = swa_model(x)

    if return_ensemble:
        return ensemble


def eval(model, test_dl, criterion, metric=None):
    device = next(model.parameters()).device

    model.eval()
    metric_value = 0

    with torch.no_grad():
        test_loss = 0
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_loss += criterion(y_pred, y)
            if metric is not None:
                metric_value += metric(y_pred, y)
        test_loss /= len(test_dl)

    if metric is not None:
        metric_value /= len(test_dl)
        return test_loss, metric_value

    return test_loss
