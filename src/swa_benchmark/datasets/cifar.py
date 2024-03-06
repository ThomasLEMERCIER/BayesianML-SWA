from torchvision.datasets import CIFAR10, CIFAR100
import torchvision


def CIFAR10Dataset(train):
    return CIFAR10(
        root="./data",
        download=True,
        transform=torchvision.transforms.ToTensor(),
        train=train,
    )


def CIFAR100Dataset(train):
    return CIFAR100(
        root="./data",
        download=True,
        transform=torchvision.transforms.ToTensor(),
        train=train,
    )
