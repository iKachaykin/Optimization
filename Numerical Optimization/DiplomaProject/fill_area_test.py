import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def indicator_function_1(x, y):
    return np.array(np.logical_and(np.logical_and(x + y >= 4, x + 4*y - 16 <= 0),
                                   11*x - 4*y - 44 <= 0), dtype=float)


def indicator_function_2(x, y):
    return np.array(x**2 + y**2 <= 25, dtype=float)


def indicator_function_3(x, y):
    return np.array(np.logical_and(np.logical_and(-x - y**2 + 4 >= 0, x >= 0),
                                   y >= 0), dtype=float)


def indicator_function_4(x, y):
    return np.array(np.logical_and(np.logical_and(np.abs(x - 1.5) <= 1.5, np.abs(y - 1.0) <= 1),
                                   x + y >= 3 / 2), dtype=float)


if __name__ == '__main__':
    colors = ('white', 'blue')
    x, y = np.linspace(-1, 6, 5000), np.linspace(-1, 6, 5000)
    xx, yy = np.meshgrid(x, y)
    z = indicator_function_1(xx, yy)
    plt.figure(figsize=(15, 7.5))
    plt.grid(True, alpha=0.5)
    plt.contourf(x, y, z, 1, cmap=ListedColormap(colors), alpha=0.2)
    plt.contour(x, y, z, 1, cmap=ListedColormap(colors[::-1]))
    plt.show()
    plt.close()
