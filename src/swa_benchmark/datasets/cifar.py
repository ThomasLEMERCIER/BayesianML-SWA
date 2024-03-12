from torchvision.datasets import CIFAR10, CIFAR100
import torchvision

torchvision.disable_beta_transforms_warning()

import torchvision.transforms.v2 as transforms
import torch


def CIFAR10Dataset(train, data_augmentation=True):
    if train and data_augmentation:
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


def CIFAR100Dataset(train, data_augmentation=True):
    if train and data_augmentation:
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
    return CIFAR100(
        root="./data",
        download=True,
        transform=transform,
        train=train,
    )
