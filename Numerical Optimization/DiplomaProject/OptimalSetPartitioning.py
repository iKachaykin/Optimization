import numpy as np
from NonLinearOptimization import trapezoid_double_on_grid_array, trapezoid_double_on_grid_matrix


# Вспомогательная функция
# Реализует преобразование пары (psi, Y) в один вектор var_max
def psi_Y_to_var_max(psi, Y):
    psi, Y = np.array(psi), np.array(Y)
    if len(psi.shape) != 1 or len(Y.shape) != 2 or psi.size != Y.shape[0]:
        raise ValueError('Input arguments are invalid!')
    return np.concatenate((psi.ravel(), Y.ravel()))


# Вспомогательная функция
# Реализует преобразование аргумента var_max в пару (psi, Y)
def var_max_to_psi_Y(var_max, partition_number):
    var_max = np.array(var_max)
    if len(var_max.shape) != 1 or var_max.size % partition_number != 0:
        raise ValueError('Input argument is invalid!')
    return var_max[:partition_number], var_max[partition_number:].reshape(partition_number, -1)


def tau_to_var_min(tau):
    tau = np.array(tau)
    return tau.ravel()


def var_min_to_tau(var_min):
    var_min = np.array(var_min)
    return var_min.reshape(2, -1)


def nonlinear_set_partitioning_target(Y, psi, tau, args):

    partition_number, product_number, cost_function, density_function, phi, phi_der, b_vector, x_left, x_right, y_left,\
        y_right, grid_dot_num_x, grid_dot_num_y = args

    if partition_number != tau.shape[1] or partition_number != psi.size or partition_number != b_vector.size or \
            partition_number != phi(Y).shape[0] or product_number != phi(Y).shape[1] or \
            phi(Y).shape[0] != phi_der(Y).shape[0] or phi(Y).shape[1] != phi_der(Y).shape[1]:
        raise ValueError('Please, check input data!')

    x, y = np.linspace(x_left, x_right, grid_dot_num_x), np.linspace(y_left, y_right, grid_dot_num_y)
    xx, yy = np.meshgrid(x, y)

    cost = cost_function(xx, yy, tau, partition_number, product_number)
    density = density_function(xx, yy, product_number)

    indicators = np.where(
        cost + psi.reshape(partition_number, 1, 1, 1) + phi_der(Y).reshape(partition_number, product_number, 1, 1) ==
        (cost + psi.reshape(partition_number, 1, 1, 1) + phi_der(Y).reshape(partition_number, product_number, 1, 1)
         ).min(axis=0), 1.0, 0.0
    )

    return (phi(trapezoid_double_on_grid_matrix(density * indicators, x_left, x_right, y_left, y_right)) +
            trapezoid_double_on_grid_matrix(cost * density * indicators, x_left, x_right, y_left, y_right)).sum()


def nonlinear_set_partitioning_target_dual(Y, psi, tau, args):

    partition_number, product_number, cost_function, density_function, phi, phi_der, b_vector, x_left, x_right, y_left,\
        y_right, grid_dot_num_x, grid_dot_num_y = args

    if partition_number != tau.shape[1] or partition_number != psi.size or partition_number != b_vector.size or \
            partition_number != phi(Y).shape[0] or product_number != phi(Y).shape[1] or \
            phi(Y).shape[0] != phi_der(Y).shape[0] or phi(Y).shape[1] != phi_der(Y).shape[1]:
        raise ValueError('Please, check input data!')

    x, y = np.linspace(x_left, x_right, grid_dot_num_x), np.linspace(y_left, y_right, grid_dot_num_y)
    xx, yy = np.meshgrid(x, y)

    cost = cost_function(xx, yy, tau, partition_number, product_number)
    density = density_function(xx, yy, product_number)

    integrand = (
            cost + psi.reshape(partition_number, 1, 1, 1) + phi_der(Y).reshape(partition_number, product_number, 1, 1)
    ).min(axis=0) * density

    return -np.dot(psi, b_vector) + (
            phi(Y) - phi_der(Y) * Y + trapezoid_double_on_grid_array(integrand, x_left, x_right, y_left, y_right)
    ).sum()


def nonlinear_set_partitioning_target_dual_with_penalties(Y, psi, tau, args, additional_args):

    partition_number, product_number, cost_function, density_function, phi, phi_der, b_vector, x_left, x_right, y_left,\
        y_right, grid_dot_num_x, grid_dot_num_y = args

    Y_penalty_left, Y_penalty_right, psi_penalty, tau_penalty, psi_constraints_indexes, tau_constraints = \
        additional_args

    return nonlinear_set_partitioning_target_dual(Y, psi, tau, args) - \
           Y_penalty_left * np.maximum(0.0, -Y).sum() - \
           Y_penalty_right * np.maximum(0.0, Y.sum(axis=1) - b_vector).sum() - \
           psi_penalty * np.maximum(0.0, -psi[psi_constraints_indexes]).sum() + \
           tau_penalty * np.maximum(0.0, tau_constraints(tau)).sum()


def nonlinear_set_partitioning_target_dual_with_penalties_ralg(var_max, var_min, args):

    sub_args, sub_additional_args = args

    partition_number, product_number, cost_function, density_function, phi, phi_der, b_vector, x_left, x_right, y_left,\
        y_right, grid_dot_num_x, grid_dot_num_y = sub_args

    Y_penalty_left, Y_penalty_right, psi_penalty, tau_penalty, psi_constraints_indexes, tau_constraints = \
        sub_additional_args

    psi, Y = var_max_to_psi_Y(var_max, partition_number)
    tau = var_min_to_tau(var_min)

    return nonlinear_set_partitioning_target_dual_with_penalties(Y, psi, tau, sub_args, sub_additional_args)


def nonlinear_set_partitioning_target_dual_temp(Y, psi, tau, args):

    partition_number, product_number, cost_function, density_function, phi, phi_der, b_vector, x_left, x_right, y_left,\
        y_right, grid_dot_num_x, grid_dot_num_y = args

    if partition_number != tau.shape[1] or partition_number != psi.size or partition_number != b_vector.size or \
            partition_number != phi(Y).shape[0] or product_number != phi(Y).shape[1] or \
            phi(Y).shape[0] != phi_der(Y).shape[0] or phi(Y).shape[1] != phi_der(Y).shape[1]:
        raise ValueError('Please, check input data!')

    x, y = np.linspace(x_left, x_right, grid_dot_num_x), np.linspace(y_left, y_right, grid_dot_num_y)
    xx, yy = np.meshgrid(x, y)

    cost = cost_function(xx, yy, tau, partition_number, product_number)
    density = density_function(xx, yy, product_number)

    integrand = (
            cost + psi.reshape(partition_number, 1, 1, 1) + phi_der(Y).reshape(partition_number, product_number, 1, 1)
    ).min(axis=0) * density

    return -np.dot(psi, b_vector) + (
            (phi(Y) - phi_der(Y) * Y).sum(axis=0) +
            trapezoid_double_on_grid_array(integrand, x_left, x_right, y_left, y_right)
    ).sum()

