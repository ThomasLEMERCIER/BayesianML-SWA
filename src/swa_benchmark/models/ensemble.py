from torch import nn
import copy
import torch


class Ensembling(nn.Module):
    def __init__(self, models: list = [], after_softmax: bool = True):
        super(Ensembling, self).__init__()
        self.models = nn.ModuleList(models)
        self.after_softmax = after_softmax

    def add_model(self, model):
        self.models.append(model)

    def add_copy(self, model):
        self.models.append(copy.deepcopy(model))

    def forward(self, x):
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")

        if self.after_softmax:
            return torch.log(
                sum(nn.functional.softmax(model(x), dim=1) for model in self.models)
                / len(self.models)
            )

        return sum(model(x) for model in self.models) / len(self.models)
