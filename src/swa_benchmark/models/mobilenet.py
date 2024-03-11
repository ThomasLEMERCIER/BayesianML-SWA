from torch import nn
import torch
from math import sqrt


class InvertedResidual(nn.Module):
    def __init__(
        self, input_channel: int, output_channel: int, stride: int, expand_ratio: int
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(input_channel * expand_ratio)
        self.use_res_connect = self.stride == 1 and input_channel == output_channel

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channel),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channel, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, output_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channel),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 216
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            [4, 32, 2, 2],
            [4, 64, 2, 2],
            [4, 96, 2, 2],
            # [4, 128, 2, 2],
            # [4, 128, 1, 1],
        ]

        # building first layer
        self.last_channel = last_channel
        self.features = [
            nn.Sequential(
                nn.Conv2d(input_size, input_channel, 3, 2, 1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True),
            )
        ]

        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = c
            for i in range(n):
                if i == 0:
                    self.features.append(
                        InvertedResidual(
                            input_channel, output_channel, s, expand_ratio=t
                        )  # type: ignore
                    )
                else:
                    self.features.append(
                        InvertedResidual(
                            input_channel, output_channel, 1, expand_ratio=t
                        )  # type: ignore
                    )
                input_channel = output_channel

        # building last several layers
        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(inplace=True),
            )
        )
        # make it nn.Sequential
        self.features_ = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, output_size)

        self._initialize_weights()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        x = self.features_(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
