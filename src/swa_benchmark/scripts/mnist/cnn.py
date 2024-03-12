import torch
from ...models import CNN
from ...utils.training import train, swa_train, test_epoch
from ...utils.scheduler import cosineLR, swaLinearLR
from ...utils.visualization import plot_loss_landspace
from ...datasets import MNISTDataset
from ...utils.metric import accuracy

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

if __name__ == "__main__":

    batch_size = 256

    ds_train, ds_test = MNISTDataset(train=True), MNISTDataset(train=False)
    print(f"Train dataset length: {len(ds_train)}, Test dataset length: {len(ds_test)}")

    train_dl = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_dl = DataLoader(
        ds_test,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    input_size = 1
    hidden_size = 32
    output_size = 10
    n_layers = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_layers=n_layers,
    ).to(device)

    epochs = 5
    eta_max = 0.01
    eta_min = 0.001

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=eta_max, weight_decay=1e-4)
    scheduler = cosineLR(
        epochs=epochs, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl)
    )
    train(
        train_dl=train_dl,
        test_dl=test_dl,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        model=model,
        epochs=epochs,
        device=device,
        metric=accuracy,
    )

    pretrained_model = CNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_layers=n_layers,
    ).to(device)
    pretrained_model.load_state_dict(model.state_dict())

    swa_model = CNN(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        n_layers=n_layers,
    ).to(device)

    swa_epochs = 12
    eta_max = 0.001
    eta_min = 0.0001
    swa_length = 3
    return_ensemble = True

    optimizer = torch.optim.SGD(model.parameters(), lr=eta_max, weight_decay=1e-4)
    swa_scheduler = swaLinearLR(
        epochs=swa_epochs,
        eta_min=eta_min,
        eta_max=eta_max,
        loader_length=len(train_dl),
        swa_epoch_length=swa_length,
    )
    _, _, ensemble_model = swa_train(
        train_dl=train_dl,
        test_dl=test_dl,
        optimizer=optimizer,
        scheduler=swa_scheduler,
        criterion=criterion,
        model=model,
        epochs=swa_epochs,
        swa_model=swa_model,
        swa_length=swa_length,
        device=device,
        metric=accuracy,
        return_ensemble=return_ensemble,
        softmax_ensemble=True,
        bn_update=True,
    )

    pretrained_loss, pretrained_metric = test_epoch(
        test_dl, pretrained_model, criterion, device, accuracy
    )
    model_loss, model_metric = test_epoch(test_dl, model, criterion, device, accuracy)
    swa_loss, swa_metric = test_epoch(test_dl, swa_model, criterion, device, accuracy)

    print("=====================================")
    print(
        f"Pretrained model test loss: {pretrained_loss:.6f}, Model test loss: {model_loss:.6f}, SWA model test loss: {swa_loss:.6f}"
    )
    print(
        f"Pretrained model test accuracy: {pretrained_metric:.6f}, Model test accuracy: {model_metric:.6f}, SWA model test accuracy: {swa_metric:.6f}"
    )
    if return_ensemble:
        ensemble_loss, ensemble_metric = test_epoch(
            test_dl, ensemble_model, criterion, device, accuracy
        )
        print(
            f"Ensemble model test loss: {ensemble_loss:.6f}, Ensemble model test accuracy: {ensemble_metric:.6f}"
        )

    print("=====================================")

    plot_loss_landspace(
        models=[pretrained_model, model, swa_model],
        criterion=criterion,
        train_dl=train_dl,
        test_dl=test_dl,
        device=device,
        n_points=5,
        point_names=["Pretrained", "Model", "SWA"],
    )
    plt.show()
