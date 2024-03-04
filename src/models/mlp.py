from torch import nn


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers):
        super(MLP, self).__init__()

        self.main = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(n_input if i == 0 else n_hidden, n_hidden),
                nn.ReLU()
            ) for i in range(n_layers)],
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x):
        return self.main(x)
