from ..utils.scheduler import constantLR, swaLinearLR
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    epochs = 200
    swa_length = 10
    swa_start = 100
    eta_max = 0.01
    eta_min = 0.001

    loader_length = 50

    training_scheduler = constantLR(epochs=swa_start, eta=eta_max, loader_length=loader_length)
    swa_scheduler = swaLinearLR(epochs=epochs-swa_start, eta_min=eta_min, eta_max=eta_max, loader_length=loader_length, swa_epoch_length=swa_length)
    scheduler = np.concatenate([training_scheduler, swa_scheduler])

    print(f"Real max learning rate: {np.max(scheduler)}, Real min learning rate: {np.min(scheduler)}")

    update_itxs = []
    for e in range(epochs):
        for it in range(loader_length):
            itx = it + e * loader_length
            continue
        
        if e > swa_start and (e+1) % swa_length == 0:
            update_itxs.append(itx)
            print(f"SWA update at iteration {itx}, learning rate: {scheduler[itx]}")

    update_itxs = np.array(update_itxs)   

    plt.plot(scheduler, label="Learning rate", marker=".")
    plt.scatter(update_itxs, scheduler[update_itxs], color="r", label="SWA update")
    plt.xlabel("Steps")
    plt.ylabel("Learning rate")
    plt.xticks(np.arange(0, (epochs+1)*loader_length, loader_length), np.arange(0, epochs+1))
    plt.grid()
    plt.title("Learning rate schedule")
    plt.show()
