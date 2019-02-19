import numpy as np
import NonLinearOptimization as nlopt
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colgen import create_colors


# Двойственный функционал задачи А6
# psi -- переменная, по которой будет происходить максимизация функционала
# tau -- переменная, по которой будет происходить минимизация функционала
# args -- аргумент функции, через который будут передаваться дополнительные параметры целевого функционала
# partition_number -- количество множеств в каждом разбиении
# product_number -- количество продуктов
# cost_function_vector -- вектор, через который передается правило вычисления стоимостей (c(x, tau))
# density_vector -- вектор, в котором содержатся плотности (rho(x))
# a_matrix -- матрица весовых коэффициентов целевого функционала
# b_vector -- вектор коэффициентов, определяющих ограничения, наложенные на подмножества из разбиений
# x_left, x_right, y_left, y_right -- границы прямоугольника, в котором можно заключить исходную область
# tau_initial -- начальное приближение центров каждого подмножества
# tau_limitations -- ограничения, наложенные на расположения центров подмножеств
# grid_dot_num -- количество узлов в сетке

def target_dual(psi, tau, args):
    tau = tau_transformation_from_vector_to_matrix(tau)

    partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right, \
    y_left, y_right, grid_dot_num_x, grid_dot_num_y = args

    if partition_number != tau.shape[1] or partition_number != psi.size or partition_number != b_vector.size or \
            partition_number != a_matrix.shape[0] or product_number != len(cost_function_vector) or \
            product_number != len(density_vector) or product_number != a_matrix.shape[1]:
        raise ValueError('Please, check input data!')

    return np.array([nlopt.trapezoid_double(lambda x, y: np.array(
        cost_function_vector[j](x, y, tau) +
        a_matrix[:, j].reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1])) +
        psi.reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1]))).min(axis=0) *
                                                         density_vector[j](x, y), x_left, x_right, y_left, y_right,
                                            grid_dot_num_x, grid_dot_num_y)
                     for j in range(product_number)]).sum() - np.dot(psi, b_vector)


# Вспомогательная функция
# Реализует преобразование аргумента tau, заданного матрицей, к вектору, путем переписывания матрицы построчно
# Возвращает вектор переписанных в однух строчку строк матрицы
def tau_transformation_from_matrix_to_vector(tau):
    tau = np.array(tau)
    if len(tau.shape) == 2:
        return tau.ravel()
    return tau.ravel().reshape(tau.shape[0], -1).T


# Вспомогательная функция
# Реализует преобразование аргумента tau, заданного вектором к матрице, исходя из того, что в векторном виде у tau
# всегда четное количество компонент; тогда результатом роботы функции будет матрица, где первая половина элементов
# вектора tau записана в первую строку матрицы, вторая половина -- во вторую
def tau_transformation_from_vector_to_matrix(tau):
    tau = np.array(tau)
    if len(tau.shape) == 1:
        return tau.reshape(2, -1)
    return tau.T.reshape(tau.shape[1], 2, -1)


# Приведение переменной оптимизации к вектору psi и матрице tau
def opt_var_to_psi_tau(opt_var):
    opt_var = np.array(opt_var)
    if opt_var.size % 3 != 0:
        raise ValueError('opt_var.size must be a multiple of 3')
    psi_tau_num = opt_var.size // 3
    psi, tau = opt_var[:psi_tau_num], np.ones((2, psi_tau_num))
    tau[0], tau[1] = opt_var[psi_tau_num:2 * psi_tau_num], opt_var[2 * psi_tau_num:3 * psi_tau_num]
    return psi, tau


# В реализованном r-алгоритме Шора аргументов целевой функции является вектор opt_var размерности n
# В задаче ОРМ искомыми величинами являются вектор psi размерности N и матрица tau размерности 2 х N
# Для использования r-алгоритма обозначим через opt_var вектор: opt_var = (psi, tau[1], tau[2]), размерности 3N,
# где скобки означают лишь то, что мы переписали вектор psi, первую строку tau (tau[1]) и вторую строку tau (tau[2])
# в одну строчку, чем и определили opt_var
# Например, если psi = (1, 2, 3)
# tau = [ [4, 5, 6],
#         [1, 2, 3]]
# то х = (1, 2, 3, 4, 5, 6, 1, 2, 3)
# psi предполагается вектором
# tau предполагается матрицей
def psi_tau_to_opt_var(psi, tau):
    return np.concatenate((psi, tau[0], tau[1]))


# Целевой функционал задачи А6
# Принимает те же аргументы, что и двойственный функционал данной задачи (target_dual)
def target(psi, tau, args):
    tau = tau_transformation_from_vector_to_matrix(tau)

    partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right, \
    y_left, y_right, grid_dot_num_x, grid_dot_num_y = args

    if partition_number != tau.shape[1] or partition_number != b_vector.size or \
            partition_number != a_matrix.shape[0] or product_number != len(cost_function_vector) or \
            product_number != len(density_vector) or product_number != a_matrix.shape[1]:
        raise ValueError('Please, check input data!')

    return np.array([
        nlopt.trapezoid_double(lambda x, y: np.array(
            (cost_function_vector[j](x, y, tau) +
             a_matrix[:, j].reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1]))) *
            np.where(
                cost_function_vector[j](x, y, tau) +
                a_matrix[:, j].reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1])) +
                psi.reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1])) ==
                np.array(cost_function_vector[j](x, y, tau) + a_matrix[:, j].reshape(partition_number, 1, 1) *
                         np.ones((partition_number, x.shape[0], x.shape[1])) +
                         psi.reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1]))
                         ).min(axis=0), 1.0, 0.0
            )
        ).sum(axis=0) * density_vector[j](x, y), x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y)
        for j in range(product_number)]).sum()


if __name__ == '__main__':

    partition_number, product_number = 30, 3

    x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y = 0.0, 12.0, 0.0, 36.0, 120, 360

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
                                     np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2),
                            lambda x, y, tau:
                            np.sqrt((x * np.ones((tau.shape[1], x.shape[0], x.shape[1])) -
                                     tau[0].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                    (y * np.ones((tau.shape[1], y.shape[0], y.shape[1])) -
                                     tau[1].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2)
                            ]

    density_vector = [lambda x, y: 1.0, lambda x, y: 1.0, lambda x, y: 1.0]

    a_matrix = np.zeros((partition_number, product_number))
    a_matrix[np.concatenate((np.arange(4), np.arange(5, 19), np.arange(20, 29))), 1] = 100.0
    a_matrix[4, 2] = 100.0

    b_vector = np.ones(partition_number) * 28.8
    b_vector[4] = 600.0
    b_vector[19] = 300.0
    b_vector[29] = 396.0

    psi_initial = np.zeros(partition_number) + 0.001
    tau_initial = np.array([
        [2.0, 6.0, 10.0, 4.0, 8.0, 2.0, 6.0, 10.0, 4.0, 8.0, 2.0, 6.0, 10.0, 4.0, 8.0, 2.0, 6.0, 10.0, 4.0, 8.0, 2.0,
         6.0, 10.0, 4.0, 8.0, 2.0, 6.0, 10.0, 4.0, 8.0],
        [1.0, 2.0, 2.0, 5.0, 5.0, 8.0, 8.0, 8.0, 11.0, 11.0, 14.0, 14.0, 14.0, 17.0, 17.0, 20.0, 20.0, 20.0, 23.0, 23.0,
         26.0, 26.0, 26.0, 29.0, 29.0, 32.0, 32.0, 32.0, 35.0, 35.0]
    ])

    psi_tau_limitations = [
        lambda psi, tau: -psi[0], lambda psi, tau: -psi[1], lambda psi, tau: -psi[2],
        lambda psi, tau: -psi[3], lambda psi, tau: -psi[5], lambda psi, tau: -psi[6],
        lambda psi, tau: -psi[7], lambda psi, tau: -psi[8], lambda psi, tau: -psi[9],
        lambda psi, tau: -psi[10], lambda psi, tau: -psi[11], lambda psi, tau: -psi[12],
        lambda psi, tau: -psi[13], lambda psi, tau: -psi[14], lambda psi, tau: -psi[15],
        lambda psi, tau: -psi[16], lambda psi, tau: -psi[17], lambda psi, tau: -psi[18],
        lambda psi, tau: -psi[20], lambda psi, tau: -psi[21], lambda psi, tau: -psi[22],
        lambda psi, tau: -psi[23], lambda psi, tau: -psi[24], lambda psi, tau: -psi[25],
        lambda psi, tau: -psi[26], lambda psi, tau: -psi[27], lambda psi, tau: -psi[28],

        lambda psi, tau: x_left - tau[0], lambda psi, tau: x_left - tau[1], lambda psi, tau: x_left - tau[2],
        lambda psi, tau: x_left - tau[3], lambda psi, tau: x_left - tau[4], lambda psi, tau: x_left - tau[5],
        lambda psi, tau: x_left - tau[6], lambda psi, tau: x_left - tau[7], lambda psi, tau: x_left - tau[8],
        lambda psi, tau: x_left - tau[9], lambda psi, tau: x_left - tau[10], lambda psi, tau: x_left - tau[11],
        lambda psi, tau: x_left - tau[12], lambda psi, tau: x_left - tau[13], lambda psi, tau: x_left - tau[14],
        lambda psi, tau: x_left - tau[15], lambda psi, tau: x_left - tau[16], lambda psi, tau: x_left - tau[17],
        lambda psi, tau: x_left - tau[18], lambda psi, tau: x_left - tau[19], lambda psi, tau: x_left - tau[20],
        lambda psi, tau: x_left - tau[21], lambda psi, tau: x_left - tau[22], lambda psi, tau: x_left - tau[23],
        lambda psi, tau: x_left - tau[24], lambda psi, tau: x_left - tau[25], lambda psi, tau: x_left - tau[26],
        lambda psi, tau: x_left - tau[27], lambda psi, tau: x_left - tau[28], lambda psi, tau: x_left - tau[29],

        lambda psi, tau: y_left - tau[30], lambda psi, tau: y_left - tau[31], lambda psi, tau: y_left - tau[32],
        lambda psi, tau: y_left - tau[33], lambda psi, tau: y_left - tau[34], lambda psi, tau: y_left - tau[35],
        lambda psi, tau: y_left - tau[36], lambda psi, tau: y_left - tau[37], lambda psi, tau: y_left - tau[38],
        lambda psi, tau: y_left - tau[39], lambda psi, tau: y_left - tau[40], lambda psi, tau: y_left - tau[41],
        lambda psi, tau: y_left - tau[42], lambda psi, tau: y_left - tau[43], lambda psi, tau: y_left - tau[44],
        lambda psi, tau: y_left - tau[45], lambda psi, tau: y_left - tau[46], lambda psi, tau: y_left - tau[47],
        lambda psi, tau: y_left - tau[48], lambda psi, tau: y_left - tau[49], lambda psi, tau: y_left - tau[50],
        lambda psi, tau: y_left - tau[51], lambda psi, tau: y_left - tau[52], lambda psi, tau: y_left - tau[53],
        lambda psi, tau: y_left - tau[54], lambda psi, tau: y_left - tau[55], lambda psi, tau: y_left - tau[56],
        lambda psi, tau: y_left - tau[57], lambda psi, tau: y_left - tau[58], lambda psi, tau: y_left - tau[59],

        lambda psi, tau: tau[0] - x_right, lambda psi, tau: tau[1] - x_right, lambda psi, tau: tau[2] - x_right,
        lambda psi, tau: tau[3] - x_right, lambda psi, tau: tau[4] - x_right, lambda psi, tau: tau[5] - x_right,
        lambda psi, tau: tau[6] - x_right, lambda psi, tau: tau[7] - x_right, lambda psi, tau: tau[8] - x_right,
        lambda psi, tau: tau[9] - x_right, lambda psi, tau: tau[10] - x_right, lambda psi, tau: tau[11] - x_right,
        lambda psi, tau: tau[12] - x_right, lambda psi, tau: tau[13] - x_right, lambda psi, tau: tau[14] - x_right,
        lambda psi, tau: tau[15] - x_right, lambda psi, tau: tau[16] - x_right, lambda psi, tau: tau[17] - x_right,
        lambda psi, tau: tau[18] - x_right, lambda psi, tau: tau[19] - x_right, lambda psi, tau: tau[20] - x_right,
        lambda psi, tau: tau[21] - x_right, lambda psi, tau: tau[22] - x_right, lambda psi, tau: tau[23] - x_right,
        lambda psi, tau: tau[24] - x_right, lambda psi, tau: tau[25] - x_right, lambda psi, tau: tau[26] - x_right,
        lambda psi, tau: tau[27] - x_right, lambda psi, tau: tau[28] - x_right, lambda psi, tau: tau[29] - x_right,

        lambda psi, tau: tau[30] - y_right, lambda psi, tau: tau[31] - y_right, lambda psi, tau: tau[32] - y_right,
        lambda psi, tau: tau[33] - y_right, lambda psi, tau: tau[34] - y_right, lambda psi, tau: tau[35] - y_right,
        lambda psi, tau: tau[36] - y_right, lambda psi, tau: tau[37] - y_right, lambda psi, tau: tau[38] - y_right,
        lambda psi, tau: tau[39] - y_right, lambda psi, tau: tau[40] - y_right, lambda psi, tau: tau[41] - y_right,
        lambda psi, tau: tau[42] - y_right, lambda psi, tau: tau[43] - y_right, lambda psi, tau: tau[44] - y_right,
        lambda psi, tau: tau[45] - y_right, lambda psi, tau: tau[46] - y_right, lambda psi, tau: tau[47] - y_right,
        lambda psi, tau: tau[48] - y_right, lambda psi, tau: tau[49] - y_right, lambda psi, tau: tau[50] - y_right,
        lambda psi, tau: tau[51] - y_right, lambda psi, tau: tau[52] - y_right, lambda psi, tau: tau[53] - y_right,
        lambda psi, tau: tau[54] - y_right, lambda psi, tau: tau[55] - y_right, lambda psi, tau: tau[56] - y_right,
        lambda psi, tau: tau[57] - y_right, lambda psi, tau: tau[58] - y_right, lambda psi, tau: tau[59] - y_right,
    ]

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, grid_dot_num_x, grid_dot_num_y)

    target_val_initial = target(psi_initial, tau_transformation_from_matrix_to_vector(tau_initial), args)
    print('tau: {0}\npsi: {1}\nTarget value: {2}'.format(tau_initial, psi_initial, target_val_initial))

    figsize = (15.0, 7.5)
    grid_dot_num_x_plotting, grid_dot_num_y_plotting = 1200, 3600
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

        plt.show()

    r_alg_results = nlopt.r_algorithm_interior_point_2_cooperative(
        target_dual, target_dual, psi_initial, tau_transformation_from_matrix_to_vector(tau_initial),
        psi_tau_limitations, psi_tau_limitations,
        lambda k: 20 * 0.85 ** k, lambda k: 20 * 0.85 ** k, target_1='max', target_2='min', args_1=args, args_2=args,
        form='H', calc_epsilon_x=1e-4, calc_epsilon_grad=1e-4, r_epsilon=16.0, iter_lim=100, print_iter_index=True,
        continue_transformation=False, step_method='adaptive_alternative',
        default_step=0.1, step_red_mult=0.1, step_incr_mult=1.25, lim_num=5, reduction_epsilon=1e-15)

    psi_solution, tau_solution = r_alg_results[0][-1], tau_transformation_from_vector_to_matrix(r_alg_results[1][-1])
    target_val_solution = target(psi_solution, tau_transformation_from_matrix_to_vector(tau_solution), args)
    print('tau: {0}\npsi: {1}\nTarget value: {2}'.format(tau_solution, psi_solution, target_val_solution))

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
