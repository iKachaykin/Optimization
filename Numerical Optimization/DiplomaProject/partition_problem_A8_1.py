import numpy as np
import NonLinearOptimization as nlopt
import OptimalSetPartitioning as osp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def cost_function(xx, yy, tau, partition_number, product_number):

    xx_array, yy_array = \
        xx * np.ones((partition_number, xx.shape[0], xx.shape[1])),\
        yy * np.ones((partition_number, yy.shape[0], yy.shape[1]))

    temp_cost = np.array([
        np.sqrt((xx_array - tau[0].reshape(partition_number, 1, 1)) ** 2 +
                (yy_array - tau[1].reshape(partition_number, 1, 1)) ** 2)
    ])

    return np.transpose(temp_cost, axes=(1, 0, 2, 3))


def cost_function_der(xx, yy, tau, partition_number, product_number):

    xx_array, yy_array = \
        xx * np.ones((partition_number, xx.shape[0], xx.shape[1])),\
        yy * np.ones((partition_number, yy.shape[0], yy.shape[1]))

    temp_cost_der = np.array([
        [
            -(xx_array - tau[0].reshape(partition_number, 1, 1)) /
            np.sqrt((xx_array - tau[0].reshape(partition_number, 1, 1)) ** 2 +
                    (yy_array - tau[1].reshape(partition_number, 1, 1)) ** 2),
            -(yy_array - tau[1].reshape(partition_number, 1, 1)) /
            np.sqrt((xx_array - tau[0].reshape(partition_number, 1, 1)) ** 2 +
                    (yy_array - tau[1].reshape(partition_number, 1, 1)) ** 2)
        ]
    ])

    return np.transpose(temp_cost_der, axes=(1, 2, 0, 3, 4))


def density_function(xx, yy, product_number):

    return np.array([
        np.where((xx - 3.0) ** 2 / 9.0 + (yy - 10.0) ** 2 / 100.0 <= 1, 1.0, 0.0)
    ])


def indicators(xx, yy, Y, psi, tau, partition_number, product_number):

    cost = cost_function(xx, yy, tau, partition_number, product_number)
    indicators = np.where(
        cost + psi.reshape(partition_number, 1, 1, 1) + phi_der(Y, A).reshape(partition_number, product_number, 1, 1) ==
        (cost + psi.reshape(partition_number, 1, 1, 1) + phi_der(Y, A).reshape(partition_number, product_number, 1, 1)
         ).min(axis=0), 1.0, 0.0
    )
    return indicators


def omega_constraints(xx, yy):
    return np.where((xx - 3.0) ** 2 / 9.0 + (yy - 10.0) ** 2 / 100.0 <= 1, 1.0, 0.0)


def tau_constraints(tau):
    return np.array([
        (tau[0] - 3.0) ** 2 / 9.0 + (tau[1] - 10.0) ** 2 / 100.0 - 1.0
    ]).T


def tau_constraints_der(tau):

    temp_der = np.array([
        [2.0 * (tau[0] - 3.0) / 9.0, 2.0 * (tau[1] - 10.0) / 100.0],
    ])

    return np.transpose(temp_der, axes=(1, 2, 0))


def phi(Y, A):
    return A * Y


def phi_der(Y, A):
    return A


def phi_second_der(Y):
    return Y - Y


if __name__ == '__main__':

    partition_number, product_number = 9, 1

    x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y = 0.0, 6.0, 0.0, 20.0, 120, 400

    b_vector = np.array([75.0, 10.0, 10.0, 10.0, 30.0, 10.0, 10.0, 10.0, 19.0])
    A = np.array([[0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0]]).T

    psi_initial = np.zeros(partition_number) + 0.0
    psi_initial = np.random.rand(partition_number) * 3
    tau_initial = np.zeros((2, partition_number)) + 0.001
    Y_initial = np.zeros((partition_number, product_number)) + 0.0

    var_max_initial, var_min_initial = osp.psi_Y_to_var_max(psi_initial, Y_initial), osp.tau_to_var_min(tau_initial)

    tau_initial[0] = np.random.rand(partition_number) * (x_right - x_left) + x_left
    tau_initial[1] = np.random.rand(partition_number) * (y_right - y_left) + y_left

    # Y_initial = np.array([
    #     [10, 100, 10, 10, 100, 10, 10, 100, 10],
    #     [10, 100, 10, 10, 100, 10, 10, 100, 10],
    #     [10, 100, 10, 10, 100, 10, 10, 100, 10]
    # ]).T

    Y_penalty_left, Y_penalty_right, psi_penalty, tau_penalty = 100000.0, 100000.0, 100000.0, 100000.0
    psi_constraints_indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

    args = (partition_number, product_number, cost_function, density_function, lambda Y: phi(Y, A),
            lambda Y: phi_der(Y, A), b_vector, x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y)

    additional_args = (Y_penalty_left, Y_penalty_right, psi_penalty, tau_penalty, psi_constraints_indexes,
                       tau_constraints)

    grad_args = (phi_second_der, cost_function_der, tau_constraints_der)

    args_ralg = (args, additional_args)

    target_val_initial = osp.nonlinear_set_partitioning_target(Y_initial, psi_initial, tau_initial, args)
    dual_target_val_initial = osp.nonlinear_set_partitioning_target_dual(Y_initial, psi_initial, tau_initial, args)
    print('Y: {0}\npsi: {1}\ntau: {2}\nTarget value: {3}\nDual target value: {4}'.format(
        Y_initial, psi_initial, tau_initial, target_val_initial, dual_target_val_initial)
    )

    scale_coeff = 0.38
    frame_x, frame_y = 1.0, 0.0
    figsize = (scale_coeff * (x_right - x_left) + frame_x, scale_coeff * (y_right - y_left) + frame_y)
    grid_dot_num_x_plotting, grid_dot_num_y_plotting = 1000, 1000
    tau_style, boundary_style = 'ko', 'k-'
    fontsize, fontweight = 14, 'bold'
    tau_text_shift = 0.06

    x_plotting, y_plotting = \
        np.linspace(x_left, x_right, grid_dot_num_x_plotting),\
        np.linspace(y_left, y_right, grid_dot_num_y_plotting)
    xx_plotting, yy_plotting = np.meshgrid(x_plotting, y_plotting)

    indicators_initial = indicators(xx_plotting, yy_plotting, Y_initial, psi_initial, tau_initial, partition_number,
                                    product_number)
    indicators_initial = indicators_initial * (np.arange(partition_number) + 1).reshape(partition_number, 1, 1, 1)

    for product in range(product_number):
        plt.figure(product, figsize=figsize)
        plt.title('Изначальное разбиение\nдля %i-го продукта' % (product + 1))
        plt.grid(True)
        plt.axis([x_left, x_right, y_left, y_right])
        plt.contour(x_plotting, y_plotting, omega_constraints(xx_plotting, yy_plotting), levels=[1.0],
                    cmap=ListedColormap(['black']))

        in_partition = []
        for partition in range(partition_number):
            if (indicators_initial[partition, product] != 0).any():
                plt.contour(x_plotting, y_plotting, indicators_initial[partition, product] *
                            np.where(
                                density_function(xx_plotting, yy_plotting, product_number)[product] != 0, 1.0, 0.0
                            ),
                            levels=[partition+1],
                            cmap=ListedColormap(['black']))
                in_partition.append(partition)
        in_partition = np.array(in_partition)

        plt.plot(tau_initial[0, in_partition], tau_initial[1, in_partition], tau_style)
        for p in in_partition:
            plt.text(tau_initial[0, p] + tau_text_shift,
                     tau_initial[1, p] + tau_text_shift,
                     '%i' % (p + 1), fontsize=fontsize, fontweight=fontweight)

    r_alg_results = nlopt.r_algorithm_cooperative(
        osp.nonlinear_set_partitioning_target_dual_with_penalties_ralg,
        osp.nonlinear_set_partitioning_target_dual_with_penalties_ralg,
        var_max_initial, var_min_initial,
        target_1='max', target_2='min', args_1=args_ralg, args_2=args_ralg,
        grad_1=lambda var_max, var_min, func, epsilon:
        -osp.nonlinear_set_partitioning_target_dual_with_penalties_grad_var_max(
            var_max, var_min, func, epsilon, args, additional_args, grad_args),
        grad_2=lambda var_max, var_min, func, epsilon:
        osp.nonlinear_set_partitioning_target_dual_with_penalties_grad_var_min(
            var_max, var_min, func, epsilon, args, additional_args, grad_args),
        form='B', beta=1/2, calc_epsilon_x=1e-7, calc_epsilon_grad=1e-7, iter_lim=1000, print_iter_index=True,
        continue_transformation=False, step_epsilon=1e-52, step_method='adaptive',
        default_step=1.0, step_red_mult=0.9, step_incr_mult=1.15, lim_num=3, reduction_epsilon=1e-15, grad_epsilon=1e-6
    )

    var_max_solution, var_min_solution = \
        r_alg_results[0][-1], r_alg_results[1][-1]
    psi_solution, Y_solution = osp.var_max_to_psi_Y(var_max_solution, partition_number)
    tau_solution = osp.var_min_to_tau(var_min_solution)

    target_val_solution = osp.nonlinear_set_partitioning_target(Y_solution, psi_solution, tau_solution, args)
    dual_target_val_solution = osp.nonlinear_set_partitioning_target_dual(Y_solution, psi_solution, tau_solution, args)

    print('Y: {0}\npsi: {1}\ntau: {2}\nTarget value: {3}\nDual target value: {4}'.format(
        Y_solution, psi_solution, tau_solution, target_val_solution, dual_target_val_solution)
    )

    indicators_solution = indicators(xx_plotting, yy_plotting, Y_solution, psi_solution, tau_solution, partition_number,
                                     product_number)
    indicators_solution = indicators_solution * (np.arange(partition_number) + 1).reshape(partition_number, 1, 1, 1)

    for product in range(product_number):
        plt.figure(product + product_number, figsize=figsize)
        plt.title('Оптимальное разбиение\nдля %i-го продукта' % (product + 1))
        plt.grid(True)
        plt.axis([x_left, x_right, y_left, y_right])
        plt.contour(x_plotting, y_plotting, omega_constraints(xx_plotting, yy_plotting), levels=[1.0],
                    cmap=ListedColormap(['black']))

        in_partition = []
        for partition in range(partition_number):
            if (indicators_solution[partition, product] != 0).any():
                plt.contour(x_plotting, y_plotting, indicators_solution[partition, product] *
                            np.where(
                                density_function(xx_plotting, yy_plotting, product_number)[product] != 0, 1.0, 0.0
                            ), levels=[partition+1],
                            cmap=ListedColormap(['black']))
                in_partition.append(partition)
        in_partition = np.array(in_partition)

        plt.plot(tau_solution[0, in_partition], tau_solution[1, in_partition], tau_style)
        for p in in_partition:
            plt.text(tau_solution[0, p] + tau_text_shift,
                     tau_solution[1, p] + tau_text_shift,
                     '%i' % (p + 1), fontsize=fontsize, fontweight=fontweight)

    plt.show()
