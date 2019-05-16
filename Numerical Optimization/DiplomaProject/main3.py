import NonLinearOptimization as nlopt
import matplotlib.pyplot as plt
from colgen import *
from numpy.linalg import norm
from scipy.optimize import rosen


def f(x):
    return norm(x, axis=0)


def alpha_k(k):
    return 3 / (k + 1)


def main():
    x0, x_left, x_right, y_left, y_right, alpha_left, alpha_right, dot_num, figsize, levels, calc_epsilon, level_max_diff, \
        exact_solution, point_seq_style, way_style, exact_solution_style, grid_alpha, min_dist_between_points = \
        [2.0, 3.0], -5.0, 5.0, -5.0, 5.0, -10.0, 10.0, 500, (15, 7.5), [], 1e-8, 0.5, [0.0, 0.0], "ko", "k-", "ro", \
        0.25, 1e-2
    plt.figure(figsize=figsize)
    points_seq = nlopt.r_algorithm(f, x0, form='B', calc_epsilon_x=calc_epsilon, iter_lim=100, step_method='adaptive',
                                   default_step=1.0, step_red_mult=0.05, step_incr_mult=2, lim_num=3,
                                   reduction_epsilon=1e-15)
    points_seq2 = nlopt.r_algorithm(f, x0, form='H', calc_epsilon_x=calc_epsilon, iter_lim=100, step_method='adaptive',
                                    default_step=1.0, step_red_mult=0.05, step_incr_mult=2, lim_num=3,
                                    reduction_epsilon=1e-15)
    print('%d - %d' % (points_seq.shape[0], points_seq2.shape[0]))
    argmin = points_seq[-1]
    min_size = np.minimum(points_seq.shape[0], points_seq2.shape[0])
    print('Отклонение между формами B и H: %f' % np.linalg.norm(points_seq[:min_size] - points_seq2[:min_size]))
    print('{0} - {1}'.format(argmin, points_seq2[-1]))
    print('--------')
    print(points_seq)
    count = 0
    while count < points_seq.shape[0] - 1:
        if norm(points_seq[count] - points_seq[count + 1]) < min_dist_between_points:
            points_seq = np.delete(points_seq, count + 1, 0)
            count -= 1
        count += 1
    points_seq = np.append(points_seq, argmin).reshape(points_seq.shape[0] + 1, 2)
    levels = np.sort(f([points_seq[:, 0], points_seq[:, 1]]))
    count = 0
    while count < levels.size - 1:
        if levels[count + 1] - levels[count] > level_max_diff:
            levels = np.insert(levels, count + 1, (levels[count + 1] + levels[count]) / 2.0)
            count -= 1
        count += 1
    levels = np.array(list(set(levels)))
    levels.sort()
    x, y = np.linspace(x_left, x_right, dot_num), np.linspace(y_left, y_right, dot_num)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = f([xx, yy])
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid("True", alpha=grid_alpha)
    colors = create_colors(levels.size)[::-1]
    plt.contour(x, y, z, levels=levels, colors=colors)
    plt.plot(points_seq[:, 0], points_seq[:, 1], point_seq_style, label=u"Наближення")
    for i in range(points_seq.shape[0] - 1):
        plt.plot([points_seq[i][0], points_seq[i + 1][0]], [points_seq[i][1], points_seq[i + 1][1]], way_style)
    plt.plot(exact_solution[0], exact_solution[1], exact_solution_style, label=u"Розв'язок")
    plt.legend(loc="upper left")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
