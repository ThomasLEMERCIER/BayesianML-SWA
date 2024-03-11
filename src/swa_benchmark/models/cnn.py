from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers):
        super(CNN, self).__init__()

        self.features = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        input_size * 2 ** (i * 2),
                        input_size * 2 ** ((i * 2) + 1),
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        input_size * 2 ** ((i * 2) + 1),
                        input_size * 2 ** ((i * 2) + 2),
                        kernel_size=3,
                        stride=1,
                        padding="same",
                    ),
                    nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                )
                for i in range(n_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_size * 2 ** (((n_layers - 1) * 2) + 2), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        features = self.features(x)
        features = F.adaptive_max_pool2d(features, (1, 1))
        features = features.view(features.size(0), -1)
        return self.classifier(features)
