from torch_geometric.datasets import GNNBenchmarkDataset

def ClusterDataset(train):
    return GNNBenchmarkDataset(root="./data", name="CLUSTER", split="train" if train else "test")
