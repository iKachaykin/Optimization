import NonLinearOptimization as nlopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def f(x):
    return x[0] + 4 * x[1]


if __name__ == '__main__':
    feasible_region_colors = ('lightgreen', 'blue', 'lightblue', 'magenta', 'cyan', 'indigo', 'orange')
    color_index = np.random.randint(0, len(feasible_region_colors))
    colors = ('white', feasible_region_colors[color_index])
    x_min, x_max, y_min, y_max = -1.0, 3.0, -1.0, 5.0
    dot_num = 10000
    figsize = (15, 7.5)
    A = np.array([[2, 4], [10, 3], [0, -1], [-1, 0]])
    b = np.array([17, 15, 0, 0])
    argmin, argmax = [0, 0], [0, 17 / 4]
    x, y = np.linspace(x_min, x_max, dot_num), np.linspace(y_min, y_max, dot_num)
    xx, yy = np.meshgrid(x, y)
    z = nlopt.feasible_region_indicator_linear(np.array([xx, yy]), A, b)
    plt.figure(figsize=figsize)
    plt.grid(True, alpha=0.5)
    plt.contourf(x, y, z, 1, cmap=ListedColormap(colors), alpha=0.2)
    plt.contour(x, y, z, 1, cmap=ListedColormap(colors[::-1]))
    levels = np.linspace(f(argmin), f(argmax), 10)
    numerical_contour = plt.contour(x, y, f([xx, yy]), levels=levels)
    plt.clabel(numerical_contour, inline=1, fontsize=10)
    plt.show()
    plt.close()
