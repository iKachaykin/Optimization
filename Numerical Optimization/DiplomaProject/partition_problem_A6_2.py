import numpy as np
import NonLinearOptimization as nlopt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colgen import create_colors


if __name__ == '__main__':

    partition_number, product_number = 9, 1

    x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y = 0.0, 6.0, 0.0, 20.0, 60, 200

    cost_function_vector = [lambda x, y, tau:
                            np.sqrt((x * np.ones((tau.shape[1], x.shape[0], x.shape[1])) -
                                     tau[0].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                    (y * np.ones((tau.shape[1], y.shape[0], y.shape[1])) -
                                     tau[1].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2)
                            ]

    density_vector = [lambda x, y: np.where((x - 3.0) ** 2 / 9.0 + (y - 10.0) ** 2 / 100.0 <= 1.0, 1.0, 0.0)]

    a_matrix = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0]).reshape(partition_number,
                                                                                             product_number)

    b_vector = np.ones(partition_number) * 10.0
    b_vector[0] = 75.0
    b_vector[4] = 30.0
    b_vector[8] = 19.0

    psi_initial = np.zeros(partition_number) + 0.001
    tau_initial = np.array([
        [3.0, 1.6, 2.9, 4.4, 5.1, 5.6, 1.0, 1.5, 3.5],
        [2.0, 1.3, 2.1, 5.7, 10.0, 11.5, 12.9, 13.9, 19.0]
    ])

    tau_initial[0] = np.random.rand(partition_number) * (x_right - x_left) + x_left
    tau_initial[1] = np.random.rand(partition_number) * (y_right - y_left) + y_left

    psi_limitations = [
        lambda psi: -psi[0], lambda psi: -psi[1], lambda psi: -psi[2],
        lambda psi: -psi[3], lambda psi: -psi[4], lambda psi: -psi[5],
        lambda psi: -psi[6], lambda psi: -psi[7], lambda psi: -psi[8]
    ]

    tau_limitations = [
        lambda tau: x_left - tau[0], lambda tau: x_left - tau[1], lambda tau: x_left - tau[2],
        lambda tau: x_left - tau[3], lambda tau: x_left - tau[4], lambda tau: x_left - tau[5],
        lambda tau: x_left - tau[6], lambda tau: x_left - tau[7], lambda tau: x_left - tau[8],

        lambda tau: y_left - tau[9], lambda tau: y_left - tau[10], lambda tau: y_left - tau[11],
        lambda tau: y_left - tau[12], lambda tau: y_left - tau[13], lambda tau: y_left - tau[14],
        lambda tau: y_left - tau[15], lambda tau: y_left - tau[16], lambda tau: y_left - tau[17],

        lambda tau: tau[0] - x_right, lambda tau: tau[1] - x_right, lambda tau: tau[2] - x_right,
        lambda tau: tau[3] - x_right, lambda tau: tau[4] - x_right, lambda tau: tau[5] - x_right,
        lambda tau: tau[6] - x_right, lambda tau: tau[7] - x_right, lambda tau: tau[8] - x_right,

        lambda tau: tau[9] - y_right, lambda tau: tau[10] - y_right, lambda tau: tau[11] - y_right,
        lambda tau: tau[12] - y_right, lambda tau: tau[13] - y_right, lambda tau: tau[14] - y_right,
        lambda tau: tau[15] - y_right, lambda tau: tau[16] - y_right, lambda tau: tau[17] - y_right
    ]

    psi_penalty, tau_penalty = 10000.0, 10000.0

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, grid_dot_num_x, grid_dot_num_y)

    additional_args = (psi_limitations, tau_limitations, psi_penalty, tau_penalty)

    target_val_initial = nlopt.linear_partition_problem_target(
        psi_initial, nlopt.tau_transformation_from_matrix_to_vector(tau_initial), args
    )
    dual_target_val_initial = nlopt.linear_partition_problem_target_dual(
        psi_initial, nlopt.tau_transformation_from_matrix_to_vector(tau_initial), args
    )
    print('tau: {0}\npsi: {1}\nTarget value: {2}\nDual target value: {3}'.format(
        tau_initial, psi_initial, target_val_initial, dual_target_val_initial)
    )

    figsize = (15.0, 7.5)
    grid_dot_num_x_plotting, grid_dot_num_y_plotting = 600, 2000
    tau_style, boundary_style = 'ko', 'k-'
    fontsize, fontweight = 14, 'bold'
    tau_text_shift = 0.06
    indicator = lambda x, y, psi, tau, j: (cost_function_vector[j](x, y, tau) +
                                           a_matrix[:, j].reshape(partition_number, 1, 1) *
                                           np.ones((partition_number, x.shape[0], x.shape[1])) +
                                           psi.reshape(partition_number, 1, 1) *
                                           np.ones((partition_number, x.shape[0], x.shape[1]))).argmin(axis=0)
    x_vals, y_vals = np.linspace(x_left, x_right, grid_dot_num_x_plotting), \
                     np.linspace(y_left, y_right, grid_dot_num_y_plotting)
    xx_grid, yy_grid = np.meshgrid(x_vals, y_vals)

    for product in range(product_number):
        plt.figure(product + 1, figsize=figsize)
        plt.title('Изначальное разбиение для %i-го продукта' % (product + 1))
        plt.grid(True)
        plt.axis([x_left, x_right, y_left, y_right])

        z = indicator(xx_grid, yy_grid, psi_initial, tau_initial, product)
        in_partition = np.unique(z)
        cf = plt.contour(x_vals, y_vals, z, levels=in_partition, cmap=ListedColormap(['black']))

        plt.plot(tau_initial[0, in_partition], tau_initial[1, in_partition], tau_style)
        for p in in_partition:
            plt.text(tau_initial[0, p] + tau_text_shift,
                     tau_initial[1, p] + tau_text_shift,
                     '%i' % (p + 1), fontsize=fontsize, fontweight=fontweight)

        # plt.colorbar(cf)

    r_alg_results = nlopt.r_algorithm_cooperative(
        lambda psi, tau, args:
        nlopt.linear_partition_problem_target_dual_interior_point(psi, tau, args, additional_args),
        lambda psi, tau, args:
        nlopt.linear_partition_problem_target_dual_interior_point(psi, tau, args, additional_args),
        psi_initial, nlopt.tau_transformation_from_matrix_to_vector(tau_initial),
        target_1='max', target_2='min', args_1=args, args_2=args,
        form='H', calc_epsilon_x=1e-4, calc_epsilon_grad=1e-10, iter_lim=1000, print_iter_index=True,
        continue_transformation=False, step_epsilon=1e-52, step_method='adaptive',
        default_step=10.0, step_red_mult=0.65, step_incr_mult=1.25, lim_num=5, reduction_epsilon=1e-15
    )

    psi_solution, tau_solution = \
        r_alg_results[0][-1], nlopt.tau_transformation_from_vector_to_matrix(r_alg_results[1][-1])
    target_val_solution = nlopt.linear_partition_problem_target(
        psi_solution, nlopt.tau_transformation_from_matrix_to_vector(tau_solution), args
    )
    dual_target_val_solution = nlopt.linear_partition_problem_target_dual(
        psi_solution, nlopt.tau_transformation_from_matrix_to_vector(tau_solution), args
    )
    print('tau: {0}\npsi: {1}\nTarget value: {2}\nDual target value: {3}'.
          format(tau_solution, psi_solution, target_val_solution, dual_target_val_solution))

    for product in range(product_number):
        plt.figure(product + product_number + 1, figsize=figsize)
        plt.title('Оптимальное разбиение для %i-го продукта' % (product + 1))
        plt.grid(True)
        plt.axis([x_left, x_right, y_left, y_right])

        z = indicator(xx_grid, yy_grid, psi_solution, tau_solution, product)
        in_partition = np.unique(z)
        cf = plt.contour(x_vals, y_vals, z, levels=in_partition, cmap=ListedColormap(['black']))

        plt.plot(tau_solution[0, in_partition], tau_solution[1, in_partition], tau_style)
        for p in in_partition:
            plt.text(tau_solution[0, p] + tau_text_shift,
                     tau_solution[1, p] + tau_text_shift,
                     '%i' % (p + 1), fontsize=fontsize, fontweight=fontweight)

        # plt.colorbar(cf)

    plt.show()

    plt.close()
