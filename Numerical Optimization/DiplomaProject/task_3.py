import NonLinearOptimization as nlopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def f(x):
    return (x[0] - 4) ** 2 + (x[1] - 5) ** 2


if __name__ == '__main__':
    feasible_region_colors = ('lightgreen', 'blue', 'lightblue', 'magenta', 'cyan', 'indigo', 'orange')
    color_index = np.random.randint(0, len(feasible_region_colors))
    colors = ('white', feasible_region_colors[color_index])
    x_min, x_max, y_min, y_max = -1.0, 6.0, -1.0, 6.0
    dot_num = 5000
    figsize = (15, 7.5)
    A = np.array([[1, 1], [-1, 0], [0, -1]])
    b = np.array([4, 0, 0])
    argmin, argmax = [1.5, 2.5], [0, 0]
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
    # tmp = nlopt.r_algorithm(lambda x: f(x) + 1 / 2 * (1 / (4 - 2 * x[0] - x[1]) + 1 / x[0] + 1 / x[1]), [0.5, 1],
    #                         iter_lim=100)
    # tmp = tmp[len(tmp) - 1]
    # print(tmp)
    # tmp_ = [0.5, 1]
    # print(f(tmp))
    # print(f(tmp_))
