import torch
from ...models import MLP
from ...utils.training import swa_train, train, test_epoch
from ...utils.visualization import plot_loss_landspace
from ...utils.scheduler import constantLR, swaLinearLR, cosineLR
from ...datasets import SyntheticDatasetFriedman

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

if __name__ == "__main__":


    n_samples = 10000
    noise = 0.1
    n_features = 4
    function = 3
    batch_size = 256

    dataset = SyntheticDatasetFriedman(n_samples=n_samples, n_features=n_features, noise=noise, function=function)
    ds_train, ds_test = random_split(dataset, [0.8, 0.2])
    print(f"Train dataset length: {len(ds_train)}, Test dataset length: {len(ds_test)}")

    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    hidden_size = 32
    output_size = 1
    n_layers = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)

    epochs = 200
    eta_max = 0.01
    eta_min = 0.0001

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=eta_max, weight_decay=1e-4)
    scheduler = cosineLR(epochs=epochs, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl))
    train(train_dl=train_dl, test_dl=test_dl, optimizer=optimizer, scheduler=scheduler, criterion=criterion, model=model, epochs=epochs, device=device)

    pretrained_model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)
    pretrained_model.load_state_dict(model.state_dict())

    swa_model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)


    epochs = 100
    eta_max = 0.00001
    eta_min = 0.0000001
    swa_length = 10

    optimizer = torch.optim.SGD(model.parameters(), lr=eta_max, weight_decay=1e-4)
    swa_scheduler = swaLinearLR(epochs=epochs, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl), swa_epoch_length=swa_length)
    swa_train(train_dl=train_dl, test_dl=test_dl, optimizer=optimizer, scheduler=swa_scheduler, criterion=criterion, model=model, epochs=epochs, swa_model=swa_model, swa_length=swa_length, device=device)

    pretrained_loss, _ = test_epoch(test_dl, pretrained_model, criterion, device)
    model_loss, _ = test_epoch(test_dl, model, criterion, device)
    swa_loss, _ = test_epoch(test_dl, swa_model, criterion, device)

    print("=====================================")
    print(f"Pretrained model test loss: {pretrained_loss:.6f}, Model test loss: {model_loss:.6f}, SWA model test loss: {swa_loss:.6f}")
    print("=====================================")

    plot_loss_landspace(models=[pretrained_model, model, swa_model], criterion=criterion, train_dl=train_dl, test_dl=test_dl, device=device, n_points=10, point_names=["Pretrained", "Model", "SWA"])
    plt.show()


