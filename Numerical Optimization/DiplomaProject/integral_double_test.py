import numpy as np
import NonLinearOptimization as nlopt
import scipy.integrate as spint
from time import time


def integrand(x, y):
    return np.ones_like(x)


if __name__ == '__main__':

    x_a, x_b, y_a, y_b, grid_dot_num_x, grid_dot_num_y = -3.0, 5.0, -3.0, 5.0, 100, 100
    x, y = np.linspace(x_a, x_b, grid_dot_num_x+1), np.linspace(y_a, y_b, grid_dot_num_y+1)
    xx, yy = np.meshgrid(x, y)
    integrand_grid = integrand(xx, yy)

    exact_val = (x_b - x_a) * (y_b - y_a)

    print('Function: %.52f\nOn grid: %.52f' %
          (nlopt.trapezoid_double(integrand, x_a, x_b, y_a, y_b, grid_dot_num_x, grid_dot_num_y),
           nlopt.trapezoid_double_on_grid(integrand_grid, x_a, x_b, y_a, y_b)))

    print('Difference between different methods: %.52f' %
          np.abs(
              nlopt.trapezoid_double(integrand, x_a, x_b, y_a, y_b, grid_dot_num_x, grid_dot_num_y) -
              nlopt.trapezoid_double_on_grid(integrand_grid, x_a, x_b, y_a, y_b)))

    print('Exact value of integral: %.52f' % exact_val)

    print('Difference between exact and numerical (on grid): %.52f' %
          np.abs(nlopt.trapezoid_double_on_grid(integrand_grid, x_a, x_b, y_a, y_b) - exact_val))
