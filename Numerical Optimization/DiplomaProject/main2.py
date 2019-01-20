import NonLinearOptimization as nlopt
import matplotlib.pyplot as plt
from colgen import *
from numpy.linalg import norm


def f(x):
    return -x[0] ** 2 - 10 * x[1] ** 2


def alpha_k(k):
    return 3 / (k + 1)


def main():
    x0, x_min, x_max, y_min, y_max, alpha_left, alpha_right, dot_num, figsize, levels, calc_epsilon, level_max_diff, \
        exact_solution, point_seq_style, way_style, exact_solution_style, grid_alpha, min_dist_between_points, \
        grads_color = \
        [5.0, 10.0], -10.0, 10.0, -10.0, 10.0, 0.0, 10.0, 500, (15, 7.5), [], 1e-4, 50, [0.0, 0.0], "ko", "k-", "ro", \
        0.25, 1e-2, "r"
    plt.figure(figsize=figsize)
    points_seq, grads_seq = nlopt.r_algorithm(f, x0, target='max', form='B', calc_epsilon_x=calc_epsilon, iter_lim=100, step_method='adaptive',
                                   default_step=10, step_red_mult=0.05, step_incr_mult=2, lim_num=3,
                                              reduction_epsilon=1e-15, return_grads=True)
    points_seq2 = nlopt.r_algorithm(f, x0, target='max', form='B', calc_epsilon=calc_epsilon, iter_lim=100, step_method='reduction',
                                    default_step=10.0, step_red_mult=0.1, reduction_epsilon=1e-15)
    argmin, last_grad = points_seq[points_seq.shape[0] - 1], grads_seq[grads_seq.shape[0] - 1]
    print('%d - %d' % (points_seq.shape[0], points_seq2.shape[0]))
    print('{0} - {1}'.format(argmin, points_seq2[points_seq2.shape[0] - 1]))
    print('--------')
    print(points_seq)
    print(grads_seq)
    count = 0
    while count < points_seq.shape[0] - 1:
        if norm(points_seq[count] - points_seq[count + 1]) < min_dist_between_points:
            points_seq = np.delete(points_seq, count + 1, 0)
            grads_seq = np.delete(grads_seq, count + 1, 0)
            count -= 1
        count += 1
    points_seq = np.append(points_seq, argmin).reshape(points_seq.shape[0] + 1, 2)
    grads_seq = np.append(grads_seq, last_grad).reshape(grads_seq.shape[0] + 1, 2)
    levels = np.sort(f([points_seq[:, 0], points_seq[:, 1]]))
    count = 0
    while count < levels.size - 1:
        if levels[count + 1] - levels[count] > level_max_diff:
            levels = np.insert(levels, count + 1, (levels[count + 1] + levels[count]) / 2.0)
            count -= 1
        count += 1
    levels = np.array(list(set(levels)))
    levels.sort()
    x, y = np.linspace(x_min, x_max, dot_num), np.linspace(y_min, y_max, dot_num)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = f([xx, yy])
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid("True", alpha=grid_alpha)
    colors = create_colors(levels.size)[::-1]
    numerical_contour = plt.contour(x, y, z, levels=levels, colors=colors)
    plt.plot(points_seq[:, 0], points_seq[:, 1], point_seq_style, label=u"Наближення")
    for i in range(points_seq.shape[0] - 1):
        plt.plot([points_seq[i][0], points_seq[i + 1][0]], [points_seq[i][1], points_seq[i + 1][1]], way_style)
    for i in range(4):
        grad_coords = -grads_seq[i]
        if norm(grad_coords) > 1:
            grad_coords = grad_coords / norm(grad_coords) / 2
        plt.arrow(points_seq[i, 0], points_seq[i, 1], grad_coords[0], grad_coords[1], color=grads_color,
                  head_width=0.1, head_length=0.1)
    plt.clabel(numerical_contour, inline=1, fontsize=10)
    plt.plot(exact_solution[0], exact_solution[1], exact_solution_style, label=u"Розв'язок")
    plt.legend(loc="upper left")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
