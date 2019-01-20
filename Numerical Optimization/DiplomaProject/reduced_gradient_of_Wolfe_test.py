import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from NonLinearOptimization import reduced_gradient_Wolfe
from colgen import create_colors


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


def remove_nearly_same_points(points, eps=1e-3):
    results = [points[0].copy()]
    for i in range(len(points) - 1):
        if np.linalg.norm(results[0] - points[i]) > eps:
            results.insert(0, points[i].copy())
    results.insert(0, points[len(points) - 1])
    return np.array(results[::-1])


def f(x):
    return (x[0] - 2) ** 4 + ((x[0] - 2) ** 2) * x[1] ** 2 + (x[1] + 1) ** 2


if __name__ == '__main__':

    feasible_region_colors = ('lightgreen', 'blue', 'lightblue', 'magenta', 'cyan', 'indigo', 'orange')
    levels_colors = ('indigo', 'darkblue', 'blue', 'green', 'lightgreen', 'orange', 'red')
    color_index = np.random.randint(0, len(feasible_region_colors))
    colors = ('white', feasible_region_colors[color_index])
    point_seq_style, exact_solution_style, way_style = 'ko', 'ro', 'k-'
    x_min, x_max, y_min, y_max = -0.25, 5.25, -0.5, 5.0
    dot_num = 10000
    figsize = (15, 7.5)
    A = np.array([[-1, -1], [1, 4], [11, -4]])
    b = np.array([-4, 16, 44])
    x0, y0 = 3.0, 3.0
    exact_solution = np.array([3.0, 1.0])
    A_modified = np.array([[-1, -1, 1, 0, 0], [1, 4, 0, 1, 0], [11, -4, 0, 0, 1]])
    initial_additional_variables = np.linalg.solve(A_modified[:, 2:], b - np.dot(A_modified[:, :2], [x0, y0]))
    initial_approximation = np.concatenate(([x0, y0], initial_additional_variables))
    points_seq = reduced_gradient_Wolfe(f, initial_approximation, A_modified, target='min', iter_lim=100000,
                                        calc_epsilon=1e-4)
    # print(len(points))
    # for point in points:
    #     print('%.4f, %.4f, %.4f, %.4f, %.4f' % (point[0], point[1], point[2], point[3], point[4]))
    points_seq = remove_nearly_same_points(points_seq)
    argmin = points_seq[len(points_seq) - 1, :2]
    print('%.16f' % np.linalg.norm(exact_solution - argmin))
    x, y = np.linspace(x_min, x_max, dot_num), np.linspace(y_min, y_max, dot_num)
    xx, yy = np.meshgrid(x, y)
    z = feasible_region_indicator_linear(np.array([xx, yy]), A, b)
    plt.figure(figsize=figsize)
    plt.grid(True, alpha=0.5)
    plt.contourf(x, y, z, 1, cmap=ListedColormap(colors), alpha=0.2)
    plt.contour(x, y, z, 1, cmap=ListedColormap(colors[::-1]))
    levels = np.concatenate(([f([5.0, 2.75])], [f([4.0, 3.0])], f([points_seq[:4, 0], points_seq[:4, 1]]),
                             [f(exact_solution)]))[::-1]
    numerical_contour = plt.contour(x, y, f([xx, yy]), levels=levels, colors=levels_colors)
    plt.clabel(numerical_contour, inline=1, fontsize=10, inline_spacing=2.0)
    plt.plot(points_seq[:, 0], points_seq[:, 1], point_seq_style, label=u"Наближення")
    for i in range(points_seq.shape[0] - 1):
        plt.plot([points_seq[i][0], points_seq[i + 1][0]], [points_seq[i][1], points_seq[i + 1][1]], way_style)
    plt.plot(exact_solution[0], exact_solution[1], exact_solution_style, label=u"Розв'язок")
    plt.legend(loc="best")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.show()
    plt.close()
