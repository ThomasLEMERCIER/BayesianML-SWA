from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(MLP, self).__init__()

        self.main = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                nn.ReLU()
            ) for i in range(n_layers)],
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.main(x)
