from torchvision.datasets import MNIST
import torchvision

def MNISTDataset(train):
    return MNIST(root='./data', download=True, transform=torchvision.transforms.ToTensor(), train=train)
