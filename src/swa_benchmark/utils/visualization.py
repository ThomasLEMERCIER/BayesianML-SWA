import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

from ..utils.training import eval

def plot_loss_landspace(device, model_1, model_2, model_3, criterion, test_dl, train_dl, z_max=10, n_points=10, point_names=[r"$w_1$", r"$w_2$", r"$w_3$"]):
    weights_1 = [p.data.detach().cpu().numpy() for p in model_1.parameters()]
    weights_2 = [p.data.detach().cpu().numpy() for p in model_2.parameters()]
    weights_3 = [p.data.detach().cpu().numpy() for p in model_3.parameters()]

    weights_shape = [w.shape if len(w.shape) > 1 else (1, *w.shape) for w in weights_1]
    weights_size = [np.prod(w) for w in weights_shape]

    weights_1 = np.concatenate([w.flatten() for w in weights_1])
    weights_2 = np.concatenate([w.flatten() for w in weights_2])
    weights_3 = np.concatenate([w.flatten() for w in weights_3])

    u = weights_2 - weights_1
    v = (weights_3 - weights_1) - (u * (np.dot(weights_3 - weights_1, u) / np.dot(u, u)))

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    first_point = (0, 0)
    second_point = (norm_u, 0)
    third_point = (np.dot(weights_3 - weights_1, u) / norm_u, norm_v)

    print(f"First point: {first_point}")
    print(f"Second point: {second_point}")
    print(f"Third point: {third_point}")

    print(f"First test: {np.max(np.abs((weights_1 - (weights_1 + first_point[0] * (u / norm_u) + first_point[1] * (v / norm_v)))))}")
    print(f"Second test: {np.max(np.abs((weights_2 - (weights_1 + second_point[0] * (u / norm_u) + second_point[1] * (v / norm_v)))))}")
    print(f"Third test: {np.max(np.abs((weights_3 - (weights_1 + third_point[0] * (u / norm_u) + third_point[1] * (v / norm_v)))))}")

    x_max = max(first_point[0], second_point[0], third_point[0])
    x_min = min(first_point[0], second_point[0], third_point[0])
    x_range = x_max - x_min

    y_max = max(first_point[1], second_point[1], third_point[1])
    y_min = min(first_point[1], second_point[1], third_point[1])
    y_range = y_max - y_min

    print(f"X max: {x_max}, X min: {x_min}, X range: {x_range}, X interval: {x_min - x_range, x_max + x_range}")
    print(f"Y max: {y_max}, Y min: {y_min}, Y range: {y_range}, Y interval: {y_min - y_range, y_max + y_range}")

    x = np.linspace(x_min - x_range*0.3, x_max + x_range*0.3, n_points)
    y = np.linspace(y_min - y_range*0.3, y_max + y_range*0.3, n_points)

    first_point_on_grid = (np.argmin(np.abs(x - first_point[0])), np.argmin(np.abs(y - first_point[1])))
    second_point_on_grid = (np.argmin(np.abs(x - second_point[0])), np.argmin(np.abs(y - second_point[1])))
    third_point_on_grid = (np.argmin(np.abs(x - third_point[0])), np.argmin(np.abs(y - third_point[1])))

    print(f"First point on grid: {first_point_on_grid}")
    print(f"Second point on grid: {second_point_on_grid}")
    print(f"Third point on grid: {third_point_on_grid}")

    xv, yv = np.meshgrid(x, y)
    z_test = np.zeros((n_points, n_points), dtype=np.float32)
    z_train = np.zeros((n_points, n_points), dtype=np.float32)

    for i in range(n_points):
        for j in range(n_points):
            print(f"Point {i*n_points + j + 1}/{n_points*n_points}", end="\r")
            w = weights_1 + xv[i, j] * (u / norm_u) + yv[i, j] * (v / norm_v)
            w = np.split(w, np.cumsum(weights_size))[:-1]
            for p, w_ in zip(model_1.parameters(), w):
                p.data = torch.tensor(w_).reshape(p.shape).to(device)   

            z_test[i, j] = eval(model_1, test_dl, criterion)
            z_train[i, j] = eval(model_1, train_dl, criterion)

    mask_test = z_test > z_max
    mask_train = z_train > z_max

    z_test[mask_test] = z_max
    z_train[mask_train] = z_max
    
    plt.subplot(1, 2, 1)
    plt.imshow(z_train, origin="lower", interpolation="lanczos", cmap=cm.jet)
    plt.scatter([first_point_on_grid[0]], [first_point_on_grid[1]], label=point_names[0])
    plt.scatter([second_point_on_grid[0]], [second_point_on_grid[1]], label=point_names[1])
    plt.scatter([third_point_on_grid[0]], [third_point_on_grid[1]], label=point_names[2])
    plt.legend()
    plt.xticks(np.linspace(0, n_points-1, 5), np.round(np.linspace(-x_range, x_range * 2, 5), 2))
    plt.yticks(np.linspace(0, n_points-1, 5), np.round(np.linspace(-y_range, y_range * 2, 5), 2))
    plt.colorbar()
    plt.title("Loss landscape (train)")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.subplot(1, 2, 2)
    plt.imshow(z_test, origin="lower", interpolation="lanczos", cmap=cm.jet)
    plt.scatter([first_point_on_grid[0]], [first_point_on_grid[1]], label=point_names[0])
    plt.scatter([second_point_on_grid[0]], [second_point_on_grid[1]], label=point_names[1])
    plt.scatter([third_point_on_grid[0]], [third_point_on_grid[1]], label=point_names[2])
    plt.legend()
    plt.xticks(np.linspace(0, n_points-1, 5), np.round(np.linspace(-x_range, x_range * 2, 5), 2))
    plt.yticks(np.linspace(0, n_points-1, 5), np.round(np.linspace(-y_range, y_range * 2, 5), 2))
    plt.colorbar()
    plt.title("Loss landscape (test)")
    plt.xlabel("u")
    plt.ylabel("v")
