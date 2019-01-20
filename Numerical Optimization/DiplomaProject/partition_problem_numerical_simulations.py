import unicodedata
import numpy as np
import NonLinearOptimization as nlopt


# В рамках данной программы предлагается решить задачу ОРМ с размещением при помощи r-алгоритма Шора
# Здесь приведен код вычислительных экспериментов, в которых начальные приближения берутся случайно


# Целевая функция поставленной задачи
# х предполагается вектором
def target_func_vector(x, args, print_flag=False):
    psi, tau_1 = x_to_psi_tau_vector(x)
    tau_2, b_2, x_a, x_b, y_a, y_b, rho, grid_dot_num = args

    c_2 = (tau_1[0] * np.ones((tau_2.shape[1], tau_1.shape[1])) -
           tau_2.T[:, 0].reshape(tau_2.shape[1], 1) * np.ones((tau_2.shape[1], tau_1.shape[1]))) ** 2 + \
          (tau_1[1] * np.ones((tau_2.shape[1], tau_1.shape[1])) -
           tau_2.T[:, 1].reshape(tau_2.shape[1], 1) * np.ones((tau_2.shape[1], tau_1.shape[1]))) ** 2
    c_2 = np.sqrt(c_2).T

    if print_flag:
        print(c_2)
        print(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0))
        print(np.dot(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0), b_2))

    # Значение интреграла, вычисленное методом трапеций при помощи описанной ниже функции
    # Аргументы передаваемой подынтегральной функции имеют несколько усложненный вид для того, чтобы подынтегральная
    # функция была определена, при переданных значениях Х и У -- сетки по х и у соответственно
    # Такой прием необходим для того, чтобы скорость вычислений была допустимой, ведь в таком случае можно использовать
    # векторизированные операции над массивами NumPy
    integral_val = trapezoid_double(lambda x, y: (np.sqrt((x * np.ones((tau_1.shape[1], x.shape[0], x.shape[1])) -
                                                           tau_1[0].reshape(tau_1.shape[1], 1, 1) *
                                                           np.ones((tau_1.shape[1], x.shape[0], x.shape[1]))) ** 2 +
                                                          (y * np.ones((tau_1.shape[1], y.shape[0], y.shape[1])) -
                                                           tau_1[1].reshape(tau_1.shape[1], 1, 1) *
                                                           np.ones((tau_1.shape[1], y.shape[0], y.shape[1]))) ** 2) +
                                                  psi.reshape(psi.size, 1, 1) *
                                                  np.ones((psi.size, x.shape[0], x.shape[1]))).min(axis=0) * rho(y, x),
                                    x_a, x_b, y_a, y_b, grid_dot_num)

    return integral_val + np.dot(np.min(c_2 - np.array([psi for _ in range(c_2.shape[1])]).T, axis=0), b_2)


# Целевая функция поставленной задачи
# х предполагается матрицей
def target_func_matrix(x, args, print_flag=False):
    psi, tau_1 = x_to_psi_tau_matrix(x)
    tau_2, b_2, x_a, x_b, y_a, y_b, rho, grid_dot_num = args

    c_2 = []
    for i in range(tau_1.shape[0]):
        t_1 = (tau_1[i][0] * np.ones((tau_2.shape[1], tau_1.shape[2])) -
               tau_2.T[:, 0].reshape(tau_2.shape[1], 1) * np.ones((tau_2.shape[1], tau_1.shape[2]))) ** 2
        t_2 = (tau_1[i][1] * np.ones((tau_2.shape[1], tau_1.shape[2])) -
               tau_2.T[:, 1].reshape(tau_2.shape[1], 1) * np.ones((tau_2.shape[1], tau_1.shape[2]))) ** 2
        t_3 = t_1 + t_2
        t_4 = np.sqrt(t_3)
        c_2.append(t_4.T)
    c_2 = np.array(c_2)

    # c_2_alt =

    # print(c_2_alt - c_2)

    a = 1

    return c_2


# Функция вычисляющая двойной интеграл методом трапеций
def trapezoid_double(integrand, x_a, x_b, y_a, y_b, N=10):
    x_vals, y_vals = np.linspace(x_a, x_b, N + 1), np.linspace(y_a, y_b, N + 1)
    xx, yy = np.meshgrid(x_vals, y_vals)
    integrand_vals = integrand(xx, yy)
    return (x_b - x_a) * (y_b - y_a) / 4 / N / N * (integrand_vals[:N, :N].sum() + integrand_vals[1:, :N].sum() +
                                                    integrand_vals[:N, 1:].sum() + integrand_vals[1:, 1:].sum())


# Приведение вектора х к вектору psi_initial и матрице tau (см. следующую функцию)
# x предполагается вектором
def x_to_psi_tau_vector(x):
    if x.size % 3 != 0:
        raise ValueError('x.size must be a multiple of 3')
    psi_tau_num = x.size // 3
    psi, tau = x[:psi_tau_num], np.ones((2, psi_tau_num))
    tau[0], tau[1] = x[psi_tau_num:2 * psi_tau_num], x[2 * psi_tau_num:3 * psi_tau_num]
    return psi, tau


# Приведение вектора х к вектору psi_initial и матрице tau (см. следующую функцию)
# x предполагается матрицей
def x_to_psi_tau_matrix(x):
    if x.shape[0] % 3 != 0:
        raise ValueError('x.shape[0] must be a multiple of 3')
    psi_tau_num = x.shape[0] // 3
    psi, tau = x[:psi_tau_num, :], np.ones((x.shape[1], 2, psi_tau_num))
    tau[:, 0] = x[psi_tau_num:2 * psi_tau_num, :].T
    tau[:, 1] = x[2 * psi_tau_num:3 * psi_tau_num, :].T
    return psi, tau


# В реализованном r-алгоритме Шора аргументов целевой функции является вектор х размерности n
# В задаче ОРМ искомыми величинами являются вектор psi_initial размерности N и матрица tau размерности 2 х N
# Для использования r-алгоритма обозначим через х вектор: х = (psi_initial, tau[1], tau[2]), размерности 3N,
# где скобки означают лишь то, что мы переписали вектор psi_initial,
# первую строку tau (tau[1]) и вторую строку tau (tau[2])
# в одну строчку, чем и определили х
# Например, если psi_initial = (1, 2, 3)
# tau = [ [4, 5, 6],
#         [1, 2, 3]]
# то х = (1, 2, 3, 4, 5, 6, 1, 2, 3)
# psi_initial предполагается вектором
# tau предполагается матрицей
def psi_tau_to_x_vector(psi, tau):
    return np.concatenate((psi, tau[0], tau[1]))


# Преобразование из psi_initial и tau обратно к х, если psi_initial -- матрица, tau -- список матриц
def psi_tau_to_x_matrix(psi, tau):
    x = np.zeros((3 * psi.shape[0], psi.shape[1]))
    x[:psi.shape[0]] = psi
    x[psi.shape[0]:2 * psi.shape[0]] = tau[:, 0].T
    x[2 * psi.shape[0]:3 * psi.shape[0]] = tau[:, 1].T
    return x


if __name__ == '__main__':

    file_path = '/Users/ivankachaikin/Documents/Учеба/Методы оптимизации/' \
                'partition_problem_numerical_simulations_output.txt'
    f = open(file_path, 'w')

    msg_start, msg_end = '\n----------------------------------\n', '\n----------------------------------\n'
    psi_ch, tau_ch = unicodedata.lookup('GREEK SMALL LETTER PSI'), unicodedata.lookup('GREEK SMALL LETTER TAU')

    sim_num = 100
    exception_num = 0

    grid_dot_num, psi_num, b_num = 1000, 5, 2
    tau_1_initial, tau_2 = \
        np.array([[0.25, 0.25, 0.75, 0.75, 0.9], [0.25, 0.75, 0.75, 0.25, 0.4]]), \
        np.array([[0.3, 0.5], [0.4, 0.8]])

    b_2 = np.array([0.2, 0.8])

    x_a, x_b = 0.0, 1.0
    y_a, y_b = 0.0, 1.0
    rho = lambda y, x: 1.0
    args = (tau_2, b_2, x_a, x_b, y_a, y_b, rho, grid_dot_num)

    g1 = [lambda x: x[5] - 1.0, lambda x: x[6] - 1.0, lambda x: x[7] - 1.0, lambda x: x[8] - 1.0,
          lambda x: x[9] - 1.0,
          lambda x: x[10] - 1.0, lambda x: x[11] - 1.0, lambda x: x[12] - 1.0, lambda x: x[13] - 1.0,
          lambda x: x[14] - 1.0]
    g2 = [lambda x: -x[5], lambda x: -x[6], lambda x: -x[7], lambda x: -x[8], lambda x: -x[9], lambda x: -x[10],
          lambda x: -x[11], lambda x: -x[12], lambda x: -x[13], lambda x: -x[14]]
    g = []
    g.extend(g1)
    g.extend(g2)

    for sim in range(sim_num):
        tau_1_initial = np.random.rand(2, psi_num)
        psi_initial = np.random.rand(psi_num) * 2 - 1
        initial_solution = psi_tau_to_x_vector(psi_initial, tau_1_initial)

        print(msg_start)
        print('Индекс вычислительного эксперимента: %d' % (sim + 1))
        print('Начальные приближения\n{0}: {1}\n{2}1:\n{3}'.format(psi_ch, psi_initial, tau_ch, tau_1_initial))
        print('Значение целевого функционала на начальном приближении: %.8f' %
              target_func_vector(initial_solution, args))

        f.write(msg_start + '\n')
        f.write('Индекс вычислительного эксперимента: %d\n' % (sim + 1))
        f.write('Начальные приближения\n{0}: {1}\n{2}1:\n{3}\n'.format(psi_ch, psi_initial, tau_ch, tau_1_initial))
        f.write('Значение целевого функционала на начальном приближении: %.8f\n' %
                target_func_vector(initial_solution, args))

        try:
            results = nlopt.r_algorithm_interior_point_2(target_func_vector, initial_solution, g, lambda k: 0.85 ** k,
                                                         args, continue_transformation=False, target='max', form='B',
                                                         tqdm_fl=False, return_grads=False, calc_epsilon_x=1e-4,
                                                         calc_epsilon_grad=1e-4, iter_lim=200,
                                                         step_method='adaptive_alternative', default_step=0.1,
                                                         step_red_mult=0.1, step_incr_mult=2, lim_num=3,
                                                         reduction_epsilon=1e-15, print_iter_index=True,
                                                         r_epsilon=1e-4)
            psi_solution, tau_1_solution = x_to_psi_tau_vector(results[-1])

            print('Оптимальное решение\n{0}: {1}\n{2}1:\n{3}'.format(psi_ch, psi_solution, tau_ch, tau_1_solution))
            print('Оптимальное значение целевого функционала: %.8f' % target_func_vector(results[-1], args))

            f.write('Оптимальное решение\n{0}: {1}\n{2}1:\n{3}\n'.format(psi_ch, psi_solution, tau_ch, tau_1_solution))
            f.write('Оптимальное значение целевого функционала: %.8f\n' % target_func_vector(results[-1], args))

        except Exception as e:
            print('Во время вычислений возникла ошибка: %s' % e.__str__())
            f.write('Во время вычислений возникла ошибка: %s\n' % e.__str__())
            exception_num += 1
        print(msg_end)
        f.write(msg_end + '\n')

    print('Количество ошибок во время вычислений: %d' % exception_num)
    f.write('Количество ошибок во время вычислений: %d\n' % exception_num)
    f.close()
