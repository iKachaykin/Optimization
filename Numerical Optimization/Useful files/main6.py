import numpy as np
import numpy.linalg as linalg
from numpy.random import rand
from tqdm import tqdm
from matplotlib import pyplot as plt
from NonLinearOptimization import r_algorithm


def f(x, args):
    a, b, c = args
    if x.shape[0] == 2:
        return 1 / 2 * (a[0, 0] * x[0] ** 2 + (a[0, 1] + a[1, 0]) * x[0] * x[1] + a[1, 1] * x[1] ** 2) - \
               b[0] * x[0] - b[1] * x[1] + c
    if len(x.shape) == 2:
        return 1 / 2 * np.diagonal(np.dot(np.dot(a, x).T, x)) - np.dot(b, x) + c * np.ones(x.shape[1])
    else:
        return 1 / 2 * np.dot(np.dot(a, x), x) - np.dot(b, x) + c


def main():
    dimension_number, exp_number = 2, 20
    x_min_dist, x_max_dist, y_min_dist, y_max_dist, x0, dot_num, points_seq_style, way_style, exact_solution_style, \
        figsize, uniform_distr_low, uniform_distr_high, calc_epsilon, min_dist_between_points, level_max_diff = \
        -10.0, 10.0, -10.0, 10.0,  [0, 0], 500, 'ko', 'k-', 'ro', (15, 7.5), -5, 5, 1e-4, 1, 1000
    a, b, c = \
        rand(dimension_number, dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low, \
        rand(dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low, 0
    for j in range(exp_number):
        b = rand(dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low
        while True:
            a = rand(dimension_number, dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low
            hessian_of_f = (a + a.T) / 2
            flag = False
            for i in range(dimension_number):
                if linalg.det(hessian_of_f[:i+1, :i+1]) < 1e-15:
                    flag = True
                    break
            if not flag:
                break
        exact_solution = linalg.solve(hessian_of_f, b)
        x0 = rand(dimension_number)
        x0[0] = x0[0] * (x_max_dist - x_min_dist) + exact_solution[0] + x_min_dist
        x0[1] = x0[1] * (y_max_dist - y_min_dist) + exact_solution[1] + y_min_dist
        points_seq, _ = r_algorithm(f, x0, args=(a, b, c), form='B', calc_epsilon=calc_epsilon, iter_lim=100,
                                step_method='adaptive', default_step=10, step_red_mult=0.75, step_incr_mult=1.25,
                                lim_num=3, reduction_epsilon=1e-15)
        argmin = points_seq[points_seq.shape[0] - 1]
        count = 0
        while count < points_seq.shape[0] - 1:
            if linalg.norm(points_seq[count] - points_seq[count + 1]) < min_dist_between_points:
                points_seq = np.delete(points_seq, count + 1, 0)
                count -= 1
            count += 1
        points_seq = np.append(points_seq, argmin).reshape(points_seq.shape[0] + 1, 2)
        levels = np.sort(f(np.array([points_seq[:, 0], points_seq[:, 1]]), (a, b, c)))
        count = 0
        while count < levels.size - 1:
            if levels[count + 1] - levels[count] > level_max_diff:
                levels = np.insert(levels, count + 1, (levels[count + 1] + levels[count]) / 2.0)
                count -= 1
            count += 1
        levels = np.array(list(set(levels)))
        levels.sort()
        x, y = \
            np.linspace(exact_solution[0] + x_min_dist, exact_solution[0] + x_max_dist, dot_num), \
            np.linspace(exact_solution[1] + y_min_dist, exact_solution[1] + y_max_dist, dot_num)
        xx, yy = np.meshgrid(x, y)
        z = f(np.array([xx, yy]), (a, b, c))
        plt.figure(figsize=figsize)
        plt.grid(True)
        numerical_contour = plt.contour(x, y, z, levels=levels)
        plt.clabel(numerical_contour, inline=1, fontsize=10)
        plt.plot(points_seq[:, 0], points_seq[:, 1], points_seq_style, label=u"Наближення")
        for i in range(points_seq.shape[0] - 1):
            plt.plot([points_seq[i][0], points_seq[i + 1][0]], [points_seq[i][1], points_seq[i + 1][1]], way_style)
        plt.plot(exact_solution[0], exact_solution[1], exact_solution_style)
        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
