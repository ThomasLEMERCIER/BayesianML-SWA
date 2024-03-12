import torch
from ...models import GraphConvolutionalNetwork as GCN
from ...utils.training import train_graph, swa_train_graph, test_epoch_graph
from ...utils.scheduler import cosineLR, swaLinearLR
from ...utils.visualization import plot_loss_landspace
from ...datasets import ClusterDataset
from ...utils.metric import accuracy

from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt


if __name__ == "__main__":

    batch_size = 16

    ds_train, ds_test = ClusterDataset(train=True), ClusterDataset(train=False)
    print(f"Train dataset length: {len(ds_train)}, Test dataset length: {len(ds_test)}")

    train_dl = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_dl = DataLoader(ds_test, batch_size=1024, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    input_size = 7
    hidden_size_graph = 32
    hidden_size_mlp = 32
    output_size = 6
    n_layers = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(input_size=input_size, hidden_size_graph=hidden_size_graph, hidden_size_mlp=hidden_size_mlp, output_size=output_size, n_layers=n_layers).to(device)

    epochs = 20
    eta_max = 0.001
    eta_min = 0.0001

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=eta_max, weight_decay=1e-4)
    scheduler = cosineLR(epochs=epochs, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl))
    train_graph(train_dl=train_dl, test_dl=test_dl, optimizer=optimizer, scheduler=scheduler, criterion=criterion, model=model, epochs=epochs, device=device, metric=accuracy)

    pretrained_model = GCN(input_size=input_size, hidden_size_graph=hidden_size_graph, hidden_size_mlp=hidden_size_mlp, output_size=output_size, n_layers=n_layers).to(device)
    pretrained_model.load_state_dict(model.state_dict())

    swa_model = GCN(input_size=input_size, hidden_size_graph=hidden_size_graph, hidden_size_mlp=hidden_size_mlp, output_size=output_size, n_layers=n_layers).to(device)

    swa_epochs = 10
    eta_max = 0.00001
    eta_min = 0.000001
    swa_length = 2

    optimizer = torch.optim.SGD(model.parameters(), lr=eta_max, weight_decay=1e-4)
    swa_scheduler = swaLinearLR(epochs=swa_epochs, eta_min=eta_min, eta_max=eta_max, loader_length=len(train_dl), swa_epoch_length=swa_length)
    swa_train_graph(train_dl=train_dl, test_dl=test_dl, optimizer=optimizer, scheduler=swa_scheduler, criterion=criterion, model=model, epochs=swa_epochs, swa_model=swa_model, swa_length=swa_length, device=device, metric=accuracy)

    pretrained_loss, pretrained_metric = test_epoch_graph(test_dl, pretrained_model, criterion, device, accuracy)
    model_loss, model_metric = test_epoch_graph(test_dl, model, criterion, device, accuracy)
    swa_loss, swa_metric = test_epoch_graph(test_dl, swa_model, criterion, device, accuracy)

    print("=====================================")
    print(f"Pretrained model test loss: {pretrained_loss:.6f}, Model test loss: {model_loss:.6f}, SWA model test loss: {swa_loss:.6f}")
    print(f"Pretrained model test accuracy: {pretrained_metric:.6f}, Model test accuracy: {model_metric:.6f}, SWA model test accuracy: {swa_metric:.6f}")
    print("=====================================")

    plot_loss_landspace(models=[pretrained_model, model, swa_model], criterion=criterion, train_dl=train_dl, test_dl=test_dl, device=device, n_points=5, point_names=["Pretrained", "Model", "SWA"], is_graph=True)
    plt.show()