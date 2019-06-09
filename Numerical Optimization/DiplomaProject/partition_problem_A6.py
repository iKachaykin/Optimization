import numpy as np
import NonLinearOptimization as nlopt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


if __name__ == '__main__':

    partition_number, product_number = 9, 2

    x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y = 0.0, 6.0, 0.0, 20.0, 60, 200

    cost_function_vector = [lambda x, y, tau:
                            np.sqrt((x * np.ones((tau.shape[1], x.shape[0], x.shape[1])) -
                                     tau[0].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                    (y * np.ones((tau.shape[1], y.shape[0], y.shape[1])) -
                                     tau[1].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2),
                            lambda x, y, tau:
                            np.sqrt((x * np.ones((tau.shape[1], x.shape[0], x.shape[1])) -
                                     tau[0].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                    (y * np.ones((tau.shape[1], y.shape[0], y.shape[1])) -
                                     tau[1].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2)
                            ]

    cost_function_vector_loop = [lambda x, y, tau_i: np.sqrt((x - tau_i[0]) ** 2 + (y - tau_i[1]) ** 2),
                                 lambda x, y, tau_i: np.sqrt((x - tau_i[0]) ** 2 + (y - tau_i[1]) ** 2)]

    d_1, d_2, xi_1, xi_2 = np.array([1.0, 5.0]), np.array([5.0, 15.0]), np.array([1.5, 4.0]), np.array([3.0, 6.0])
    density_vector = [lambda x, y: 100.0 / (2.0 * np.pi * xi_1[0] * xi_1[1]) *
                                   np.exp(-((x - d_1[0]) ** 2 / xi_1[0] ** 2 + (y - d_1[1]) ** 2 / xi_1[1] ** 2) / 2),
                      lambda x, y: 100.0 / (2.0 * np.pi * xi_2[0] * xi_2[1]) *
                                   np.exp(-((x - d_2[0]) ** 2 / xi_2[0] ** 2 + (y - d_2[1]) ** 2 / xi_2[1] ** 2) / 2)]

    a_matrix = np.ones((partition_number, product_number)) * 100.0
    a_matrix[0, 0] = 0.0
    a_matrix[8, 0] = 0.0
    a_matrix[0, 1] = 0.0
    a_matrix[4, 1] = 0.0
    a_matrix[7, 1] = 0.0

    b_vector = np.ones(partition_number) * 20
    b_vector[0] = 150.0
    b_vector[4] = 60.0
    b_vector[8] = 38.0

    psi_initial, tau_initial = np.zeros(partition_number) + 0.001, np.zeros((2, partition_number)) + 0.001

    tau_initial[0] = np.random.rand(partition_number) * (x_right - x_left) + x_left
    tau_initial[1] = np.random.rand(partition_number) * (y_right - y_left) + y_left

    # tau_initial[:, 0] = np.array([2.0, 8.1])
    # tau_initial[:, 4] = np.array([3.8, 17.6])
    # tau_initial[:, 7] = np.array([3.9, 13.0])
    # tau_initial[:, 8] = np.array([1.4, 3.2])

    psi_penalty, tau_penalty = 10000.0, 10000.0
    psi_limitations_inds = np.arange(partition_number)

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, grid_dot_num_x, grid_dot_num_y)

    args_loop = (partition_number, product_number, cost_function_vector_loop, density_vector, a_matrix, b_vector,
                 x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y)

    additional_args = (psi_penalty, psi_limitations_inds, tau_penalty)

    target_val_initial = nlopt.linear_partition_problem_target(
        psi_initial, nlopt.tau_transformation_from_matrix_to_vector(tau_initial), args
    )
    dual_target_val_initial = nlopt.linear_partition_problem_target_dual(
        psi_initial, nlopt.tau_transformation_from_matrix_to_vector(tau_initial), args
    )
    print('tau: {0}\npsi: {1}\nTarget value: {2}\nDual target value: {3}'.format(
        tau_initial, psi_initial, target_val_initial, dual_target_val_initial)
    )

    # target_val_initial_loop = nlopt.linear_partition_problem_target_loop(
    #     psi_initial, nlopt.tau_transformation_from_matrix_to_vector(tau_initial), args_loop
    # )
    # print(target_val_initial_loop)
    #
    # dual_target_val_initial_loop = nlopt.linear_partition_problem_target_dual_loop(
    #     psi_initial, nlopt.tau_transformation_from_matrix_to_vector(tau_initial), args_loop
    # )
    # print(dual_target_val_initial_loop)

    scale_coeff = 0.385
    frame_x, frame_y = 1.65, 0.0
    figsize = (scale_coeff * (x_right-x_left) + frame_x, scale_coeff * (y_right-y_left) + frame_y)
    grid_dot_num_x_plotting, grid_dot_num_y_plotting = 600, 2000
    tau_style, boundary_style = 'ko', 'k-'
    fontsize, fontweight = 14, 'bold'
    tau_text_shift = 0.06
    indicator = lambda x, y, psi, tau, j: np.where(density_vector[j](x, y) != 0,
                                                   (cost_function_vector[j](x, y, tau) +
                                                    a_matrix[:, j].reshape(partition_number, 1, 1) *
                                                    np.ones((partition_number, x.shape[0], x.shape[1])) +
                                                    psi.reshape(partition_number, 1, 1) *
                                                    np.ones((partition_number, x.shape[0], x.shape[1]))).argmin(axis=0),
                                                   -1)
    x_vals, y_vals = np.linspace(x_left, x_right, grid_dot_num_x_plotting),\
                     np.linspace(y_left, y_right, grid_dot_num_y_plotting)
    xx_grid, yy_grid = np.meshgrid(x_vals, y_vals)

    for product in range(product_number):
        plt.figure(product + 1, figsize=figsize)
        plt.title('Изначальное разбиение для %i-го продукта' % (product + 1))
        plt.grid(True)
        plt.axis([x_left, x_right, y_left, y_right])

        z = indicator(xx_grid, yy_grid, psi_initial, tau_initial, product)
        in_partition = np.unique(z)
        in_partition = in_partition[in_partition >= 0]
        cf = plt.contour(x_vals, y_vals, z, levels=in_partition, cmap=ListedColormap(['black']))
        plt.contour(x_vals, y_vals, density_vector[product](xx_grid, yy_grid), levels=[0.0],
                    cmap=ListedColormap(['black']))

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
        form='B', calc_epsilon_x=1e-4, calc_epsilon_grad=1e-10, iter_lim=1000, print_iter_index=True,
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

    # target_val_solution_loop = nlopt.linear_partition_problem_target_loop(
    #     psi_solution, nlopt.tau_transformation_from_matrix_to_vector(tau_solution), args_loop
    # )
    # print(target_val_solution_loop)
    #
    # dual_target_val_solution_loop = nlopt.linear_partition_problem_target_dual_loop(
    #     psi_solution, nlopt.tau_transformation_from_matrix_to_vector(tau_solution), args_loop
    # )
    # print(dual_target_val_solution_loop)

    for product in range(product_number):
        plt.figure(product + product_number + 1, figsize=figsize)
        plt.title('Оптимальное разбиение для %i-го продукта' % (product + 1))
        plt.grid(True)
        plt.axis([x_left, x_right, y_left, y_right])

        z = indicator(xx_grid, yy_grid, psi_solution, tau_solution, product)
        in_partition = np.unique(z)
        in_partition = in_partition[in_partition >= 0]
        cf = plt.contour(x_vals, y_vals, z, levels=in_partition, cmap=ListedColormap(['black']))
        plt.contour(x_vals, y_vals, density_vector[product](xx_grid, yy_grid), levels=[0.0],
                    cmap=ListedColormap(['black']))

        in_partition = np.unique(z)
        plt.plot(tau_solution[0, in_partition], tau_solution[1, in_partition], tau_style)
        for p in in_partition:
            plt.text(tau_solution[0, p] + tau_text_shift,
                     tau_solution[1, p] + tau_text_shift,
                     '%i' % (p + 1), fontsize=fontsize, fontweight=fontweight)

    plt.show()

    plt.close()
