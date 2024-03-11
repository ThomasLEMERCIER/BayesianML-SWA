import torch
from ...models import MLP
from ...utils.training import swa_train, train, eval
from ...utils.visualization import plot_loss_landspace
from ...utils.scheduler import constantLR, swaLinearLR, cosineLR
from ...datasets import SyntheticDataset2D

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import matplotlib.pyplot as plt

if __name__ == "__main__":


    n_samples = 100 # the final dataset will have n_samples * n_samples samples
    n_features = 2 # the final dataset will have n_features features
    interval = (-2, 2) # the final dataset will have features in the interval (-2, 2)
    noise = 0.1 # the final dataset will have noise with a standard deviation of 0.1

    batch_size = 32

    hidden_size = 16
    output_size = 1
    n_layers = 3

    dataset = SyntheticDataset2D(n_samples, interval, noise)
    ds_train, ds_test = random_split(dataset, [0.8, 0.2])
    print(f"Train dataset length: {len(ds_train)}, Test dataset length: {len(ds_test)}")

    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)

    epochs = 100
    eta_max = 0.001
    eta_min = 0.00005

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta_max, weight_decay=1e-4)
    scheduler = cosineLR(epochs=epochs, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl))
    train(model, train_dl, test_dl, criterion, optimizer, epochs, scheduler)

    pretrained_model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)
    pretrained_model.load_state_dict(model.state_dict())

    swa_model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)

    epochs = 100
    swa_length = 20
    swa_start = 0

    swa_scheduler = swaLinearLR(epochs=epochs-swa_start, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl), swa_epoch_length=swa_length)
    swa_train(model, swa_model, train_dl, test_dl, criterion, optimizer, epochs, swa_length, swa_start, swa_scheduler)


    pretrained_model_loss = eval(pretrained_model, test_dl, criterion)
    model_loss = eval(model, test_dl, criterion)
    swa_loss = eval(swa_model, test_dl, criterion)

    print(f"Pretrained model test loss: {pretrained_model_loss:.4f}, Model test loss: {model_loss:.4f}, SWA model test loss: {swa_loss:.4f}")

    plot_loss_landspace(device, pretrained_model, model, swa_model, criterion, test_dl, train_dl, point_names=["Pretrained", "Model", "SWA"])
    plt.show()

