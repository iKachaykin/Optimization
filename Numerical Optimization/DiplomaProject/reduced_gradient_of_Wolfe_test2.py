import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from NonLinearOptimization import reduced_gradient_Wolfe


def feasible_region_indicator_linear(x, A, b):
    result = np.array(np.ones_like(A[0, 0] * x[0]), dtype=bool)
    for i in range(A.shape[0]):
        sum = 0
        for j in range(A.shape[1]):
            sum += A[i, j] * x[j]
        result = np.logical_and(sum <= b[i], result)
    return np.array(result, dtype=float)


def feasible_region_indicator(x, g):
    result = np.array(np.ones_like(g[0](x)), dtype=bool)
    for i in range(len(g)):
        result = np.logical_and(g[i](x) <= 0, result)
    return np.array(result, dtype=float)


def f(x):
    return 2 * x[0] ** 2 + 2 * x[1] ** 2 - 2 * x[0] * x[1] - 4 * x[0] - 6 * x[1]


if __name__ == '__main__':
    colors = ('white', 'blue')
    x_min, x_max, y_min, y_max = -1.0, 6.0, -1.0, 6.0
    dot_num = 5000
    figsize = (15, 7.5)
    A = np.array([[1, 1], [1, 5], [-1, 0], [0, -1]])
    b = np.array([2, 5, 0, 0])
    x0, y0 = 0.0, 0.0
    A_modified = np.array([[1, 1, 1, 0], [1, 5, 0, 1]])
    initial_approximation = [x0, y0, 2.0, 5.0]
    print(reduced_gradient_Wolfe(f, initial_approximation, A_modified,  iter_lim=1000, calc_epsilon=1e-6))
    x, y = np.linspace(x_min, x_max, dot_num), np.linspace(y_min, y_max, dot_num)
    xx, yy = np.meshgrid(x, y)
    z = feasible_region_indicator_linear(np.array([xx, yy]), A, b)
    plt.figure(figsize=figsize)
    plt.grid(True, alpha=0.5)
    plt.contourf(x, y, z, 1, cmap=ListedColormap(colors), alpha=0.2)
    plt.contour(x, y, z, 1, cmap=ListedColormap(colors[::-1]))
    plt.show()
    plt.close()
