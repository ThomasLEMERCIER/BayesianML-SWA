from .swa import update_swa
import torch
import time

def train(model, train_dl, test_dl, criterion, optimizer, epochs, scheduler):
    device = next(model.parameters()).device

    test_loss = 0
    for e in range(epochs):
        model.train()
        start = time.time()
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

        test_loss = eval(model, test_dl, criterion)

        print(f"Epoch {e+1}/{epochs}, Test loss: {test_loss:.4f}, Time: {time.time() - start:.2f} s")
    
    return test_loss

def swa_train(model, swa_model, train_dl, test_dl, criterion, optimizer, epochs, swa_length, swa_start, scheduler):
    swa_n = 0
    device = next(model.parameters()).device

    test_loss = 0
    for e in range(epochs):
        model.train()
        start = time.time()
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

        if e > swa_start and (e+1) % swa_length == 0:
            update_swa(swa_model, model, swa_n)
            swa_n += 1

        test_loss = eval(model, test_dl, criterion)

        print(f"Epoch {e+1}/{epochs}, Test loss: {test_loss:.4f}, Time: {time.time() - start:.2f} s, {' (SWA update) with learning rate at: ' + str(optimizer.param_groups[0]['lr'].item()) if e > swa_start and (e+1) % swa_length == 0 else ''}")
    
    return test_loss

def eval(model, test_dl, criterion):
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        test_loss = 0
        for x, y in test_dl:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            test_loss += criterion(y_pred, y)
        test_loss /= len(test_dl)
    return test_loss
