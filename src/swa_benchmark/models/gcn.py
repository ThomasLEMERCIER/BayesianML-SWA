from torch_geometric.nn.models import GCN
import torch

class GraphConvolutionalNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size_graph, hidden_size_mlp, n_layers):
        super(GraphConvolutionalNetwork, self).__init__()
        self.gcn = GCN(in_channels=input_size, hidden_channels=hidden_size_graph, out_channels=hidden_size_graph, num_layers=n_layers)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size_graph, hidden_size_mlp),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size_mlp, output_size)
        )

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.mlp(x)
        return x

