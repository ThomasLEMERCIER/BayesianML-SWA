import torch
from ...models import MLP
from ...utils.training import train, eval
from ...utils.scheduler import constantLR, cosineLR
from ...utils.visualization import plot_loss_landspace
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

    dataset = SyntheticDataset2D(n_samples, n_features, interval, noise)
    ds_train, ds_test = random_split(dataset, [0.8, 0.2])
    print(f"Train dataset length: {len(ds_train)}, Test dataset length: {len(ds_test)}")

    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_1 = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)
    model_2 = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)
    model_3 = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)

    epochs = 20
    eta_max = 0.01
    eta_min = 0.0001

    criterion = torch.nn.MSELoss()
    optimizer_1 = torch.optim.SGD(model_1.parameters(), lr=eta_max, weight_decay=1e-4)
    optimizer_2 = torch.optim.SGD(model_2.parameters(), lr=eta_max, weight_decay=1e-4)
    optimizer_3 = torch.optim.SGD(model_3.parameters(), lr=eta_max, weight_decay=1e-4)

    drift_scheduler = constantLR(epochs=1, eta=0.1, loader_length=len(train_dl))
    training_scheduler = cosineLR(epochs=epochs-1, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl))
    scheduler = torch.tensor([*drift_scheduler, *training_scheduler])

    train(model_1, train_dl, test_dl, criterion, optimizer_1, epochs, scheduler)
    train(model_2, train_dl, test_dl, criterion, optimizer_2, epochs, scheduler)
    train(model_3, train_dl, test_dl, criterion, optimizer_3, epochs, scheduler)

    loss_1 = eval(model_1, test_dl, criterion)
    loss_2 = eval(model_2, test_dl, criterion)
    loss_3 = eval(model_3, test_dl, criterion)

    print(f"Model 1 test loss: {loss_1:.4f}, Model 2 test loss: {loss_2:.4f}, Model 3 test loss: {loss_3:.4f}")
    plot_loss_landspace(device, model_1, model_2, model_3, criterion, test_dl, train_dl)
    plt.show()
