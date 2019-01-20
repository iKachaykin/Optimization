import NonLinearOptimization as ralg
import numpy as np
import scipy.integrate as spint
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def target_func_old(psi, args, print_flag=False):
    tau_1, tau_2, b_2, x_a, x_b, y_a, y_b, rho = args
    c_2 = np.empty((tau_1.shape[1], tau_2.shape[1]))
    for i in range(c_2.shape[0]):
        for j in range(c_2.shape[1]):
            c_2[i, j] = np.sqrt((tau_1[0, i] - tau_2[0, j]) ** 2 + (tau_1[1, i] - tau_2[1, j]) ** 2)
    if print_flag:
        print(c_2)
        print(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0))
        print(np.dot(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0), b_2))
    return spint.dblquad(lambda y, x: np.min(np.sqrt((x - tau_1[0]) ** 2 + (y - tau_1[1]) ** 2) + psi) * rho(y, x),
                         x_a, x_b, y_a, y_b)[0] + \
           np.dot(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0), b_2)


def target_func_new(psi, args, print_flag=False):
    tau_1, tau_2, b_2, x_a, x_b, y_a, y_b, rho, grid_dot_num = args
    c_2 = np.empty((tau_1.shape[1], tau_2.shape[1]))
    for i in range(c_2.shape[0]):
        for j in range(c_2.shape[1]):
            c_2[i, j] = np.sqrt((tau_1[0, i] - tau_2[0, j]) ** 2 + (tau_1[1, i] - tau_2[1, j]) ** 2)
    if print_flag:
        print(c_2)
        print(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0))
        # print(np.dot(np.min(c_2 - np.array([psi_initial for _ in range(c_2.shape[1])]).T, axis=0), b_2))
    return trapezoid_double(lambda x, y: (np.sqrt((x * np.ones((tau_1.shape[1], x.shape[0], x.shape[1])) -
                                                   tau_1[0].reshape(tau_1.shape[1], 1, 1) *
                                                   np.ones((tau_1.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                                  (y * np.ones((tau_1.shape[1], y.shape[0], y.shape[1])) -
                                                   tau_1[1].reshape(tau_1.shape[1], 1, 1) *
                                                   np.ones((tau_1.shape[1], y.shape[0], y.shape[1]))) ** 2) +
                                          psi.reshape(psi.size, 1, 1) *
                                          np.ones((psi.size, x.shape[0], x.shape[1]))).min(axis=0) * rho(y, x),
                            x_a, x_b, y_a, y_b, grid_dot_num) +\
           np.dot(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0), b_2)


def target_grad_old(psi, args):
    tau_1, tau_2, b_2, x_a, x_b, y_a, y_b, rho = args
    c_2 = np.empty((tau_1.shape[1], tau_2.shape[1]))
    for i in range(c_2.shape[0]):
        for j in range(c_2.shape[1]):
            c_2[i, j] = np.sqrt((tau_1[0, i] - tau_2[0, j]) ** 2 + (tau_1[1, i] - tau_2[1, j]) ** 2)
    grad = np.empty_like(psi)
    for i in range(grad.size):
        grad[i] = spint.dblquad(lambda y, x: (1.0
                                              if np.argmin(np.sqrt((x - tau_1[0]) ** 2 + (y - tau_1[1]) ** 2) + psi) ==
                                                 i
                                              else 0.0) * rho(y, x), x_a, x_b, y_a, y_b)[0] + \
                  np.dot(
                      np.where(i == np.argmin(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0),
                               -1.0, 0.0),
                      b_2)
    return grad


def target_grad_new(psi, args):
    tau_1, tau_2, b_2, x_a, x_b, y_a, y_b, rho, grid_dot_num = args
    c_2 = np.empty((tau_1.shape[1], tau_2.shape[1]))
    for i in range(c_2.shape[0]):
        for j in range(c_2.shape[1]):
            c_2[i, j] = np.sqrt((tau_1[0, i] - tau_2[0, j]) ** 2 + (tau_1[1, i] - tau_2[1, j]) ** 2)
    grad = np.empty_like(psi)
    print('--------------------------------------')
    for i in range(grad.size):
        grad[i] = trapezoid_double(lambda x, y: np.where(
            (np.sqrt((x * np.ones((tau_1.shape[1], x.shape[0], x.shape[1])) -
                      tau_1[0].reshape(tau_1.shape[1], 1, 1) * np.ones((tau_1.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                     (y * np.ones((tau_1.shape[1], y.shape[0], y.shape[1])) -
                      tau_1[1].reshape(tau_1.shape[1], 1, 1) *
                      np.ones((tau_1.shape[1], y.shape[0], y.shape[1]))) ** 2) +
             psi.reshape(psi.size, 1, 1) * np.ones((psi.size, x.shape[0], x.shape[1]))).min(axis=0) ==
            np.sqrt((x - tau_1[0, i]) ** 2 + (y - tau_1[1, i]) ** 2) + psi[i],
            1.0, 0.0) * rho(y, x), x_a, x_b, y_a, y_b, grid_dot_num) +\
                  np.dot(np.where(np.abs(c_2[i] - psi[i] -
                                         np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0)) < 1e-10,
                                  -1.0, 0.0), b_2)
        print(np.dot(np.where(np.abs(c_2[i] - psi[i] -
                                         np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0)) < 1e-10,
                                  -1.0, 0.0), b_2))
    print('---------------------------------------')
    return grad


def trapezoid_double(integrand, x_a, x_b, y_a, y_b, N=10):
    x_vals, y_vals = np.linspace(x_a, x_b, N+1), np.linspace(y_a, y_b, N+1)
    xx, yy = np.meshgrid(x_vals, y_vals)
    integrand_vals = integrand(xx, yy)
    return (x_b - x_a) * (y_b - y_a) / 4 / N / N * (integrand_vals[:N, :N].sum() + integrand_vals[1:, :N].sum() +
                                                    integrand_vals[:N, 1:].sum() + integrand_vals[1:, 1:].sum())


if __name__ == '__main__':

    first_to_print = 3
    figsize = (7.5, 7.5)
    fontsize = 14
    fontweight = 'bold'
    grid_dot_num, psi_num, b_num = 1000, 5, 2
    tau_1, tau_2 = \
        np.array([[0.25, 0.25, 0.75, 0.75, 0.9], [0.25, 0.75, 0.75, 0.25, 0.4]]),\
        np.array([[0.3, 0.5], [0.4, 0.8]])
    b_2 = np.array([0.2, 0.8])
    psi = np.zeros(psi_num)
    x_a, x_b = 0.0, 1.0
    y_a, y_b = 0.0, 1.0
    rho = lambda y, x: 1.0
    args = (tau_1, tau_2, b_2, x_a, x_b, y_a, y_b, rho, grid_dot_num)
    print(target_func_new(psi, args))
    lst_with_ticks = ['']
    lst_with_ticks.extend([str(i) for i in range(1, psi_num + 1)])
    x, y = np.linspace(x_a, x_b, grid_dot_num), np.linspace(y_a, y_b, grid_dot_num)
    xx, yy = np.meshgrid(x, y)
    indicator = lambda x, y, psi: (np.sqrt((x * np.ones((tau_1.shape[1], x.shape[0], x.shape[1])) -
                                       tau_1[0].reshape(tau_1.shape[1], 1, 1) *
                                       np.ones((tau_1.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                      (y * np.ones((tau_1.shape[1], y.shape[0], y.shape[1])) -
                                       tau_1[1].reshape(tau_1.shape[1], 1, 1) *
                                       np.ones((tau_1.shape[1], y.shape[0], y.shape[1]))) ** 2) +
                              psi.reshape(psi.size, 1, 1) *
                              np.ones((psi.size, x.shape[0], x.shape[1]))).argmin(axis=0) + 1.0
    # t1, t2 = 0.0, 0.0
    # t1_av, t2_av = 0.0, 0.0
    # sim_num = 10
    # for psi_initial in np.random.rand(sim_num, psi_num) * 2 - 1:
    #     t1 = time.clock()
    #     g1 = nlopt.middle_grad_non_matrix(psi_initial, lambda x: target_func_vector(x, args))
    #     t1 = time.clock() - t1
    #     t1_av += t1
    #     t2 = time.clock()
    #     g2 = target_grad_new(psi_initial, args)
    #     t2 = time.clock() - t2
    #     t2_av += t2
    #     print(g1, '\n', g2, '\n', np.linalg.norm(g1-g2), '\n--------------------------------\n')
    # t1_av, t2_av = t1_av / sim_num, t2_av / sim_num
    # print('Average time:\n', t1_av, ' -- ', t2_av)
    # print(nlopt.middle_grad_non_matrix(psi_initial, lambda x: target_func_vector(x, args)), '\n',
    #       target_grad_new(psi_initial, args), '\n')
    results, grads = ralg.r_algorithm(target_func_new, psi, args,
                                      # grad=lambda x0, func, epsilon: target_grad_new(x0, args),
                                      continue_transformation=False,
                                      target='max', tqdm_fl=False, return_grads=True,
                                      calc_epsilon_x=1e-4, calc_epsilon_grad=1e-4, iter_lim=500,
                                      step_method='adaptive_alternative', default_step=0.1, step_red_mult=0.1,
                                      step_incr_mult=1.25, lim_num=5, reduction_epsilon=1e-15)
    for i in range(first_to_print):
        z = indicator(xx, yy, results[i])
        fig0 = plt.figure(i, figsize=figsize)
        # contour0 = plt.contourf(x, y, z, np.arange(0, psi_num + 1))
        # colorbar0 = fig0.colorbar(contour0)
        # colorbar0.set_ticklabels(lst_with_ticks)
        plt.contour(x, y, z, cmap=ListedColormap(['black']))
        for j in range(psi_num):
            plt.text(tau_1[0, j], tau_1[1, j], '%i' % (j + 1), fontsize=fontsize, fontweight=fontweight)
    psi_solution, grad_solution = results[-1], grads[-1]
    print(results[:first_to_print], '\n...\n', psi_solution,
          '\n-------------------------------------------------\n', grads[:first_to_print], '\n...\n', grad_solution)
    for r in results[:first_to_print]:
        print(target_func_new(r, args))
    print('...\n', target_func_new(psi_solution, args))
    z = indicator(xx, yy, psi_solution)
    fig1 = plt.figure(first_to_print, figsize=figsize)
    plt.contour(x, y, z, cmap=ListedColormap(['black']))
    for j in range(psi_num):
        plt.text(tau_1[0, j], tau_1[1, j], '%i' % (j + 1), fontsize=fontsize, fontweight=fontweight)
    # contour1 = plt.contourf(x, y, z, np.arange(0, psi_num+1))
    # colorbar1 = fig1.colorbar(contour1)
    # colorbar1.set_ticklabels(lst_with_ticks)
    plt.show()
    plt.close()
