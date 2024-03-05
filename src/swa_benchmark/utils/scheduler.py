import numpy as np

def constantLR(epochs, eta, loader_length):
    return np.full(epochs * loader_length, eta)

def cosineLR(epochs, eta_min, eta_max, loader_length):
    T_max = epochs * loader_length
    steps = np.arange(0, T_max)

    lr = eta_min + 0.5 * (eta_max - eta_min) * (
        1
        + np.cos(steps / T_max * np.pi)
    )

    return lr

def linearLR(epochs, eta_min, eta_max, loader_length):
    steps = np.arange(0, epochs * loader_length)

    lr = eta_max + (eta_min - eta_max) * steps / (epochs * loader_length)

    return lr

def swaLinearLR(epochs, eta_min, eta_max, loader_length, swa_epoch_length):
    steps = np.arange(0, swa_epoch_length * loader_length)

    lr_swa = eta_max + (eta_min - eta_max) * steps / (swa_epoch_length * loader_length)
    lr = np.tile(lr_swa, epochs // swa_epoch_length)

    return lr

def swaCosineLR(epochs, eta_min, eta_max, loader_length, swa_epoch_length):
    T_max = swa_epoch_length * loader_length
    steps = np.arange(0, T_max)

    lr_swa = eta_min + 0.5 * (eta_max - eta_min) * (
        1
        + np.cos(steps / T_max * np.pi)
    )
    lr = np.tile(lr_swa, epochs // swa_epoch_length)

    return lr
