from torch.utils.data import Dataset
import torch

from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3


class SyntheticDataset2D(Dataset):
    def __init__(self, n_samples, interval=(-1, 1), noise=0.1):
        self.n_samples = n_samples * n_samples

        linear_space = torch.linspace(interval[0], interval[1], n_samples)
        self.xv, self.yv = torch.meshgrid(linear_space, linear_space, indexing="ij")

        self.x = torch.stack([self.xv, self.yv], dim=-1).view(-1, 2)
        self.y = (
            2 * torch.cos(torch.pi * self.x[:, 0] * self.x[:, 1])
            - 0.5 * self.x[:, 0]
            + 0.08 * self.x[:, 1]
            + torch.sin(torch.pi * self.x[:, 0])
            - torch.sin(torch.pi * self.x[:, 1])
            + noise * torch.randn(self.n_samples)
        )
        self.y = self.y.view(-1, 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class SyntheticDatasetFriedman(Dataset):
    def __init__(self, n_samples, n_features, noise=0.1, function=1):
        self.n_samples = n_samples
        if function == 1:
            if n_features < 4:
                raise ValueError("n_features must be at least 4 for function 1")
            self.x, self.y = make_friedman1(n_samples=n_samples, n_features=n_features, noise=noise)
        elif function == 2:
            if n_features != 4:
                raise ValueError("n_features must be 4 for function 2")
            self.x, self.y = make_friedman2(n_samples=n_samples, noise=noise)
        elif function == 3:
            if n_features != 4:
                raise ValueError("n_features must be 4 for function 3")
            self.x, self.y = make_friedman3(n_samples=n_samples, noise=noise)
        else:
            raise ValueError("function must be 1, 2 or 3")
        
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
