import numpy as np
import NonLinearOptimization as nlopt
import scipy.integrate as spint
from time import time


def integrand(x, y):
    return np.ones_like(x)


if __name__ == '__main__':

    x_a, y_a, x_step, y_step = 0.0, 0.0, 0.1, 0.075
    x_b_min, x_b_max, y_b_min, y_b_max, grid_dot_num_x, grid_dot_num_y = 1.0, 2.0, 1.0, 2.0, 1000, 1000

    integrals_exact, integrals_numerical = [], []
    integrand_grid = []

    all_x_a, all_x_b, all_y_a, all_y_b = [], [], [], []

    for x_b in np.arange(x_b_min, x_b_max, x_step):

        all_x_a.append([])
        all_x_b.append([])
        all_y_a.append([])
        all_y_b.append([])

        integrand_grid.append([])
        integrals_exact.append([])

        for y_b in np.arange(y_b_min, y_b_max, y_step):

            all_x_a[-1].append(x_a)
            all_x_b[-1].append(x_b)
            all_y_a[-1].append(y_a)
            all_y_b[-1].append(y_b)

            x, y = np.linspace(x_a, x_b, grid_dot_num_x+1), np.linspace(y_a, y_b, grid_dot_num_y+1)
            xx, yy = np.meshgrid(x, y)
            integrand_grid[-1].append(integrand(xx, yy))

            integrals_exact[-1].append((x_b - x_a) * (y_b - y_a))

    integrand_grid = np.array(integrand_grid)

    all_x_a, all_x_b, all_y_a, all_y_b = np.array(all_x_a), np.array(all_x_b), np.array(all_y_a), np.array(all_y_b)

    integrals_exact = np.array(integrals_exact)
    integrals_numerical = nlopt.trapezoid_double_on_grid_matrix(integrand_grid, all_x_a, all_x_b, all_y_a, all_y_b)

    print('Exact values of integral:\n{0}'.format(integrals_exact))
    print('Numerical values of integral:\n{0}'.format(integrals_numerical))

    print('Difference between exact and numerical (on grid): %f' %
          np.linalg.norm(integrals_exact - integrals_numerical))
