import numpy as np
import NonLinearOptimization as nlopt


# Функция вычисляющая двойной интеграл методом трапеций
def trapezoid_double(integrand, x_left, x_right, y_left, y_right, N=10):
    x_vals, y_vals = np.linspace(x_left, x_right, N + 1), np.linspace(y_left, y_right, N + 1)
    xx, yy = np.meshgrid(x_vals, y_vals)
    integrand_vals = integrand(xx, yy)
    return (x_right - x_left) * (y_right - y_left) / 4 / N / N *\
           (integrand_vals[:N, :N].sum() + integrand_vals[1:, :N].sum() +
            integrand_vals[:N, 1:].sum() + integrand_vals[1:, 1:].sum())


# Преобразование tau, заданное в форме матрицы, в вектор
def tau_matrix_to_vector(tau):
    return tau.ravel()


# Преобразование tau, заданное в форме вектора, в матрицу
def tau_vector_to_matrix(tau):
    return tau.reshape(2, -1)


# Функция расстояний от точек множества к центрам подмножеств
def c_func(x, y, tau):
    np.sqrt((x * np.ones((tau.shape[1], x.shape[0], x.shape[1])) -
             tau[0].reshape(tau.shape[1], 1, 1) * np.ones((tau.shape[1], x.shape[0], x.shape[1]))) ** 2 +
            (y * np.ones((tau.shape[1], y.shape[0], y.shape[1])) -
             tau[1].reshape(tau.shape[1], 1, 1) * np.ones((tau.shape[1], y.shape[0], y.shape[1]))) ** 2)


# Целевая функция
def target(psi, args):
    c_func, g_tau, partition_num, exact_lim_num, tau_initial, a_vect, b_vect, x_left, x_right, y_left, y_right,\
    rho, grid_dot_num = args
    subtarget = lambda tau: trapezoid_double(lambda x, y:
                                             (c_func(x, y, tau_vector_to_matrix(tau)) +
                                              psi.reshape(partition_num, 1, 1) *
                                              np.ones((partition_num, x.shape[0], x.shape[1])) +
                                              a_vect.reshape(partition_num, 1, 1) *
                                              np.ones((partition_num, x.shape[0], x.shape[1]))).min(axis=0) * rho(x, y),
                                             x_left, x_right, y_left, y_right, grid_dot_num) - np.dot(psi, b_vect)
    tau_min = nlopt.r_algorithm_interior_point_1(subtarget, tau_matrix_to_vector(tau_initial), g_tau,
                                                 lambda k: 0.85 ** k)[-1]
    return trapezoid_double(lambda x, y: (c_func(x, y, tau_min) +
                                          psi.reshape(partition_num, 1, 1) *
                                          np.ones((partition_num, x.shape[0], x.shape[1])) +
                                          a_vect.reshape(partition_num, 1, 1) *
                                          np.ones((partition_num, x.shape[0], x.shape[1]))).min(axis=0) * rho(x, y),
                            x_left, x_right, y_left, y_right, grid_dot_num) - np.dot(psi, b_vect)


if __name__ == '__main__':

    grid_dot_num = 200
    partition_num, exact_lim_num = 9, 0
    x_left, x_right, y_left, y_right = 0, 6, 0, 20

    g_tau = [lambda tau: x_left - tau[0], lambda tau: x_left - tau[1], lambda tau: x_left - tau[2],
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
             lambda tau: tau[15] - y_right, lambda tau: tau[16] - y_right, lambda tau: tau[17] - y_right]

    g_psi = [lambda psi: -psi[0], lambda psi: -psi[1], lambda psi: -psi[2],
             lambda psi: -psi[3], lambda psi: -psi[4], lambda psi: -psi[5],
             lambda psi: -psi[6], lambda psi: -psi[7], lambda psi: -psi[8]]

    tau_initial = np.array([
        [3.0, 1.6, 2.9, 4.4, 5.1, 5.6, 1.0, 1.5, 3.5],
        [2.0, 1.3, 2.1, 5.7, 10.0, 11.5, 12.9, 13.9, 19.0]
    ])
    psi_initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    a_vect = np.array([0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 0.0])
    b_vect = np.array([75.0, 10.0, 10.0, 10.0, 30.0, 10.0, 10.0, 10.0, 19.0])

    rho = lambda x, y: 1.0 if (x - 3.0) * (x - 3.0) / 9 + (y - 10.0) * (y - 10.0) / 100 <= 1 else 0.0

    args = (c_func, g_tau, partition_num, exact_lim_num, tau_initial, a_vect, b_vect, x_left, x_right, y_left, y_right,
           rho, grid_dot_num)

    print(target(psi_initial, args))
