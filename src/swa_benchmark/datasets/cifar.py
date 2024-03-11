from torchvision.datasets import CIFAR10, CIFAR100
import torchvision
import torchvision.transforms.v2 as transforms
import torch


def CIFAR10Dataset(train):
    if train:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToImageTensor(),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )
    else:
        transform = transforms.Compose(
            [transforms.ToImageTensor(), transforms.ConvertImageDtype(torch.float32)]
        )
    return CIFAR10(
        root="./data",
        download=True,
        transform=transform,
        train=train,
    )


def CIFAR100Dataset(train):
    return CIFAR100(
        root="./data",
        download=True,
        transform=torchvision.transforms.ToTensor(),
        train=train,
    )
