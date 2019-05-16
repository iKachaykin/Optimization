import numpy as np
import NonLinearOptimization as nlopt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


if __name__ == '__main__':

    partition_number, product_number = 16, 1

    x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y = 0.0, 1.0, 0.0, 1.0, 200, 200

    cost_function_vector = [lambda x, y, tau:
                            np.sqrt((x * np.ones((tau.shape[1], x.shape[0], x.shape[1])) -
                                     tau[0].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                    (y * np.ones((tau.shape[1], y.shape[0], y.shape[1])) -
                                     tau[1].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2)
                            ]

    density_vector = [lambda x, y: 1.0 * (x*x + 1) / (x*x + 1)]

    a_matrix = np.zeros((partition_number, product_number))

    b_vector = np.zeros(partition_number)

    psi_initial = np.zeros(partition_number)

    tau_initial = np.zeros((2, partition_number))

    tau_initial[0] = np.random.rand(partition_number) * (x_right - x_left) + x_left
    tau_initial[1] = np.random.rand(partition_number) * (y_right - y_left) + y_left

    psi_penalty, tau_penalty = 1.0, 100000.0
    psi_limitations_inds = np.arange(partition_number)

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, grid_dot_num_x, grid_dot_num_y)

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

    scale_coeff = 7.0
    frame_x, frame_y = 0.0, 0.0
    figsize = (scale_coeff * (x_right-x_left) + frame_x, scale_coeff * (y_right-y_left) + frame_y)
    grid_dot_num_x_plotting, grid_dot_num_y_plotting = 1000, 1000
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

    r_alg_results = nlopt.r_algorithm(
        lambda tau, args:
        nlopt.linear_partition_problem_target_dual_interior_point(psi_initial, tau, args, additional_args),
        nlopt.tau_transformation_from_matrix_to_vector(tau_initial), target='min', args=args,
        form='B', calc_epsilon_x=1e-4, calc_epsilon_grad=1e-4, iter_lim=1000, print_iter_index=True,
        continue_transformation=False, step_epsilon=1e-52, step_method='adaptive',
        default_step=1.01, step_red_mult=0.1, step_incr_mult=1.1, lim_num=3, reduction_epsilon=1e-15
    )

    psi_solution, tau_solution = \
        psi_initial.copy(), nlopt.tau_transformation_from_vector_to_matrix(r_alg_results[-1])
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
        in_partition = in_partition[in_partition >= 0]
        cf = plt.contour(x_vals, y_vals, z, levels=in_partition, cmap=ListedColormap(['black']))
        plt.contour(x_vals, y_vals, density_vector[product](xx_grid, yy_grid), levels=[0.0],
                    cmap=ListedColormap(['black']))

        plt.plot(tau_solution[0, in_partition], tau_solution[1, in_partition], tau_style)
        for p in in_partition:
            plt.text(tau_solution[0, p] + tau_text_shift,
                     tau_solution[1, p] + tau_text_shift,
                     '%i' % (p + 1), fontsize=fontsize, fontweight=fontweight)

        # plt.colorbar(cf)

    plt.show()

    plt.close()
