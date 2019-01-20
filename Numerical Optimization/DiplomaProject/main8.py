import numpy as np
import NonLinearOptimization as nlopt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def f(x):
    return 2 * (x[0] + 1) ** 2 + (x[1] + 1) ** 2


if __name__ == '__main__':

    g = [lambda x: -x[0], lambda x: -x[1]]
    feasible_region_colors = ('lightgreen', 'blue', 'lightblue', 'magenta', 'cyan', 'indigo', 'orange')
    levels_colors = ('indigo', 'darkblue', 'blue', 'green', 'lightgreen', 'orange', 'red')
    color_index = np.random.randint(0, len(feasible_region_colors))
    colors = ('white', feasible_region_colors[color_index])
    point_seq_style, exact_solution_style, way_style = 'ko', 'ro', 'k-'
    x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0
    dot_num = 1000
    figsize = (15, 7.5)
    exact_solution = np.array([0.0, 0.0])
    initial_solution = np.array([1.0, 1.0])

    points_seq = nlopt.r_algorithm_interior_point_2(f, initial_solution, g, lambda k: 0.85 ** k,
                                                    grad=nlopt.middle_grad, calc_epsilon_x=1e-9, calc_epsilon_grad=1e-4,
                                                    step_epsilon=1e-8,
                                                    r_epsilon=1e-4, iter_lim=100, continue_transformation=False,
                                                    step_method='adaptive', default_step=0.1,
                                                    step_red_mult=0.25, step_incr_mult=1.1, lim_num=5,
                                                    reduction_epsilon=1e-15)
    argmin = points_seq[-1]
    print(argmin)
    points_seq = nlopt.remove_nearly_same_points(points_seq)
    print('%.16f' % np.linalg.norm(exact_solution - argmin))

    x, y = np.linspace(x_min, x_max, dot_num), np.linspace(y_min, y_max, dot_num)
    xx, yy = np.meshgrid(x, y)
    z = nlopt.feasible_region_indicator(np.array([xx, yy]), g)

    plt.figure(figsize=figsize)
    plt.grid(True, alpha=0.5)
    plt.contourf(x, y, z, 1, cmap=ListedColormap(colors), alpha=0.2)
    plt.contour(x, y, z, 1, cmap=ListedColormap(colors[::-1]))
    plt.plot(points_seq[:, 0], points_seq[:, 1], point_seq_style, label=u"Наближення")
    for i in range(points_seq.shape[0] - 1):
        plt.plot([points_seq[i][0], points_seq[i + 1][0]], [points_seq[i][1], points_seq[i + 1][1]], way_style)
    plt.plot(exact_solution[0], exact_solution[1], exact_solution_style, label=u"Розв'язок")
    plt.legend(loc="best")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.show()
    plt.close()
