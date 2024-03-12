import torch


def accuracy(output, target):
    """Computes the accuracy metric for a given batch of predictions."""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = output.argmax(dim=1)
        correct = pred.eq(target).sum()
        acc = correct.float() / batch_size

    return acc.item()
