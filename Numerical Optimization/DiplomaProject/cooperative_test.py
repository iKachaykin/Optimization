import NonLinearOptimization as nlopt
import numpy as np


def func_1(x, y):
    return (x**2).sum() + (y**2).sum()


def func_2(x, y):
    return (x**2).sum() + (y**2).sum()


if __name__ == '__main__':

    x0, y0 = np.ones(10) - 5, np.ones(54) * 2

    print(x0)
    print(y0)

    x_sol, y_sol = nlopt.r_algorithm_cooperative(func_1, func_2, x0, y0, None, None, nlopt.middle_grad_non_matrix_pool,
                                                 nlopt.middle_grad_non_matrix_pool, form='B', target_2='min',
                                                 calc_epsilon_x=1e-4, iter_lim=100, print_iter_index=True,
                                                 step_method='adaptive_alternative', default_step=0.1,
                                                 step_red_mult=0.1, step_incr_mult=2, lim_num=3)

    print(x_sol[-1])
    print(y_sol[-1])
