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

    partition_number, product_number = 10, 1

    x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y = 0.0, 1.0, 0.0, 1.0, 200, 200

    cost_function_vector = [lambda x, y, tau:
                            np.sqrt((x * np.ones((tau.shape[1], x.shape[0], x.shape[1])) -
                                     tau[0].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                    (y * np.ones((tau.shape[1], y.shape[0], y.shape[1])) -
                                     tau[1].reshape(tau.shape[1], 1, 1) *
                                     np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2)
                            ]

    density_vector = [lambda x, y: 1.0]

    a_matrix = np.ones((partition_number, product_number)) * 100.0

    b_vector = np.ones(partition_number) * 10.0

    psi_initial = np.zeros(partition_number) + 1.0

    tau_initial = np.empty((2, partition_number))
    tau_initial[0] = np.random.rand(partition_number) * (x_right - x_left) + x_left
    tau_initial[1] = np.random.rand(partition_number) * (y_right - y_left) + y_left

    psi_tau_limitations = [
        lambda psi, tau: -psi[0], lambda psi, tau: -psi[1], lambda psi, tau: -psi[2],
        lambda psi, tau: -psi[3], lambda psi, tau: -psi[4], lambda psi, tau: -psi[5],
        lambda psi, tau: -psi[6], lambda psi, tau: -psi[7], lambda psi, tau: -psi[8],
        lambda psi, tau: -psi[9],

        lambda psi, tau: x_left - tau[0], lambda psi, tau: x_left - tau[1], lambda psi, tau: x_left - tau[2],
        lambda psi, tau: x_left - tau[3], lambda psi, tau: x_left - tau[4], lambda psi, tau: x_left - tau[5],
        lambda psi, tau: x_left - tau[6], lambda psi, tau: x_left - tau[7], lambda psi, tau: x_left - tau[8],
        lambda psi, tau: x_left - tau[9],

        lambda psi, tau: y_left - tau[10], lambda psi, tau: y_left - tau[11], lambda psi, tau: y_left - tau[12],
        lambda psi, tau: y_left - tau[13], lambda psi, tau: y_left - tau[14], lambda psi, tau: y_left - tau[15],
        lambda psi, tau: y_left - tau[16], lambda psi, tau: y_left - tau[17], lambda psi, tau: y_left - tau[18],
        lambda psi, tau: y_left - tau[19],

        lambda psi, tau: tau[0] - x_right, lambda psi, tau: tau[1] - x_right, lambda psi, tau: tau[2] - x_right,
        lambda psi, tau: tau[3] - x_right, lambda psi, tau: tau[4] - x_right, lambda psi, tau: tau[5] - x_right,
        lambda psi, tau: tau[6] - x_right, lambda psi, tau: tau[7] - x_right, lambda psi, tau: tau[8] - x_right,
        lambda psi, tau: tau[9] - x_right,

        lambda psi, tau: tau[10] - y_right, lambda psi, tau: tau[11] - y_right, lambda psi, tau: tau[12] - y_right,
        lambda psi, tau: tau[13] - y_right, lambda psi, tau: tau[14] - y_right, lambda psi, tau: tau[15] - y_right,
        lambda psi, tau: tau[16] - y_right, lambda psi, tau: tau[17] - y_right, lambda psi, tau: tau[18] - y_right,
        lambda psi, tau: tau[19] - y_right
    ]

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, 10 * grid_dot_num_x, 10 * grid_dot_num_y)

    target_val_initial = target(psi_initial, tau_transformation_from_matrix_to_vector(tau_initial), args)
    target_dual_val_initial = target_dual(psi_initial, tau_transformation_from_matrix_to_vector(tau_initial), args)
    print('tau: {0}\npsi: {1}\nTarget value: {2}\nDual target value: {3}'.
          format(tau_initial, psi_initial, target_val_initial, target_dual_val_initial))

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, grid_dot_num_x, grid_dot_num_y)

    figsize = (15.0, 7.5)
    grid_dot_num_x_plotting, grid_dot_num_y_plotting = 1000, 1000
    tau_style, boundary_style = 'ko', 'k-'
    fontsize, fontweight = 14, 'bold'
    tau_text_shift = 0.005
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

    r_alg_results = nlopt.r_algorithm_interior_point_4_cooperative(
        target_dual, target_dual, psi_initial, tau_transformation_from_matrix_to_vector(tau_initial),
        psi_tau_limitations, psi_tau_limitations,
        lambda k: 20 * 0.85 ** k, lambda k: 20 * 0.85 ** k, target_1='max', target_2='min', args_1=args, args_2=args,
        form='H', calc_epsilon_x=1e-4, calc_epsilon_grad=1e-4, r_epsilon=4.0, iter_lim=10000, print_iter_index=True,
        continue_transformation=False, step_method='adaptive',
        default_step=0.1, step_red_mult=0.1, step_incr_mult=1.25, lim_num=5, reduction_epsilon=1e-15)

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, 10 * grid_dot_num_x, 10 * grid_dot_num_y)

    psi_solution, tau_solution = r_alg_results[0][-1], tau_transformation_from_vector_to_matrix(r_alg_results[1][-1])
    target_val_solution = target(psi_solution, tau_transformation_from_matrix_to_vector(tau_solution), args)
    target_dual_val_solution = target_dual(psi_solution, tau_transformation_from_matrix_to_vector(tau_solution), args)
    print('tau: {0}\npsi: {1}\nTarget value: {2}\nDual target value: {3}'.
          format(tau_solution, psi_solution, target_val_solution, target_dual_val_solution))

    args = (partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right,
            y_left, y_right, grid_dot_num_x, grid_dot_num_y)

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
