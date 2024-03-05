import torch
from ..models import MLP
from ..utils.training import train, eval
from ..utils.scheduler import constantLR, swaLinearLR
from ..datasets import SyntheticDataset

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

if __name__ == "__main__":


    n_samples = 100 # the final dataset will have n_samples * n_samples samples
    n_features = 2 # the final dataset will have n_features features
    interval = (-2, 2) # the final dataset will have features in the interval (-2, 2)
    noise = 0.1 # the final dataset will have noise with a standard deviation of 0.1

    batch_size = 32

    hidden_size = 16
    output_size = 1
    n_layers = 3

    dataset = SyntheticDataset(n_samples, n_features, interval, noise)
    ds_train, ds_test = random_split(dataset, [0.8, 0.2])
    print(f"Train dataset length: {len(ds_train)}, Test dataset length: {len(ds_test)}")

    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)
    swa_model = MLP(input_size=n_features, hidden_size=hidden_size, output_size=output_size, n_layers=n_layers).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}, Training on {device}")

    epochs = 100
    swa_length = 20
    swa_start = 20
    eta_max = 0.01
    eta_min = 0.0001

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta_max, momentum=0.9, weight_decay=1e-4)

    training_scheduler = constantLR(epochs=swa_start, eta=eta_max, loader_length=len(train_dl))
    swa_scheduler = swaLinearLR(epochs=epochs-swa_start, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl), swa_epoch_length=swa_length)
    scheduler = torch.tensor([*training_scheduler, *swa_scheduler])

    train(model, swa_model, train_dl, test_dl, criterion, optimizer, epochs, swa_length, swa_start, scheduler)


    swa_loss = eval(swa_model, test_dl, criterion)
    model_loss = eval(model, test_dl, criterion)

    print(f"SWA model test loss: {swa_loss:.4f}, Model test loss: {model_loss:.4f}")
    print(f"SWA model test loss is {'better' if swa_loss < model_loss else 'worse'} than the model test loss")

    if n_features == 2:
        import matplotlib.pyplot as plt
        from matplotlib import cm

        z_pred = model(dataset.x.to(device)).cpu()
        z_pred = z_pred.detach().numpy().reshape(n_samples, n_samples)

        z_pred_swa = swa_model(dataset.x.to(device)).cpu()
        z_pred_swa = z_pred_swa.detach().numpy().reshape(n_samples, n_samples)

        fig, ax = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))
        surf = ax[0].plot_surface(dataset.xv, dataset.yv, z_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax[0].view_init(50, 45)
        ax[0].set_title(f"Model (MSE: {model_loss:.4f})")
        fig.colorbar(surf, shrink=0.5, aspect=5)

        surf = ax[1].plot_surface(dataset.xv, dataset.yv, z_pred_swa, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax[1].view_init(50, 45)
        ax[1].set_title(f"SWA Model (MSE: {swa_loss:.4f})")
        fig.colorbar(surf, shrink=0.5, aspect=5)

        surf = ax[2].plot_surface(dataset.xv, dataset.yv, dataset.y.reshape(n_samples, n_samples), cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax[2].view_init(50, 45)
        ax[2].set_title("True")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
