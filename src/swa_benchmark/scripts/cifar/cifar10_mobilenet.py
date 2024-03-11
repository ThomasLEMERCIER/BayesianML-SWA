import torch
from ...models import MobileNetV2
from ...utils.training import train, swa_train, eval
from ...utils.scheduler import cosineLR, swaLinearLR
from ...utils.visualization import plot_loss_landspace
from ...datasets import CIFAR100Dataset, CIFAR10Dataset
from ...utils.metric import accuracy


from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

if __name__ == "__main__":

    batch_size = 256

    ds_train, ds_test = CIFAR10Dataset(train=True), CIFAR10Dataset(train=False)
    print(f"Train dataset length: {len(ds_train)}, Test dataset length: {len(ds_test)}")

    train_dl = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_dl = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    input_size = 3
    output_size = 10
    return_ensemble = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV2(
        input_size=input_size,
        output_size=output_size,
    ).to(device)

    epochs = 50
    eta_max = 0.001
    eta_min = 0.00005

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta_max, weight_decay=1e-4)
    scheduler = cosineLR(
        epochs=epochs, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl)
    )
    train(
        model,
        train_dl,
        test_dl,
        criterion,
        optimizer,
        epochs,
        scheduler,
        metric=accuracy,
    )

    pretrained_model = MobileNetV2(
        input_size=input_size,
        output_size=output_size,
    ).to(device)
    pretrained_model.load_state_dict(model.state_dict())

    swa_model = MobileNetV2(
        input_size=input_size,
        output_size=output_size,
    ).to(device)

    epochs = 15
    swa_length = 5
    swa_start = 0

    swa_scheduler = swaLinearLR(
        epochs=epochs - swa_start,
        eta_min=eta_min,
        eta_max=eta_max,
        loader_length=len(train_dl),
        swa_epoch_length=swa_length,
    )
    ensemble = swa_train(
        model,
        swa_model,
        train_dl,
        test_dl,
        criterion,
        optimizer,
        epochs,
        swa_length,
        swa_start,
        swa_scheduler,
        return_ensemble=return_ensemble,
        metric=accuracy,
    )

    pretrained_model_loss, pretrained_metric = eval(
        pretrained_model, test_dl, criterion, metric=accuracy
    )
    model_loss, model_metric = eval(model, test_dl, criterion, metric=accuracy)
    swa_loss, swa_metric = eval(swa_model, test_dl, criterion, metric=accuracy)

    print(
        f"Pretrained model test loss: {pretrained_model_loss:.4f}, Model test loss: {model_loss:.4f}, SWA model test loss: {swa_loss:.4f}"
    )
    print(
        f"Pretrained model test metric: {pretrained_metric:.4f}, Model test metric: {model_metric:.4f}, SWA model test metric: {swa_metric:.4f}"
    )
    if return_ensemble:
        ensemble_loss, ensemble_metric = eval(
            ensemble, test_dl, criterion, metric=accuracy
        )
        print(
            f"Ensemble test loss: {ensemble_loss:.4f}, Ensemble test metric: {ensemble_metric:.4f}"
        )

    plot_loss_landspace(
        device,
        pretrained_model,
        model,
        swa_model,
        criterion,
        test_dl,
        train_dl,
        point_names=["Pretrained", "Model", "SWA"],
        n_points=5,
    )
    plt.show()
