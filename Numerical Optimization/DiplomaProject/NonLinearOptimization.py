import numpy as np
import numpy.linalg as linalg
from scipy.misc import derivative
from math import isnan
from tqdm import tqdm as tqdm
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from numpy.polynomial import legendre as leg


def dichotomy(func, a, b, a_lst=None, b_lst=None, target="min", epsilon=1e-10, iter_lim=1000000):
    if np.abs(epsilon) < 2e-15:
        raise ValueError("epsilon is to small to perform calculations")
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    counter, delta = 0, epsilon / 2
    if a_lst is not None:
        a_lst.append(a)
    if b_lst is not None:
        b_lst.append(b)
    while b - a > epsilon and counter < iter_lim:
        if sign * func((a + b - delta) / 2.0) <= sign * func((a + b + delta) / 2.0):
            b = (a + b + delta) / 2.0
        else:
            a = (a + b - delta) / 2.0
        if a_lst is not None:
            a_lst.append(a)
        if b_lst is not None:
            b_lst.append(b)
        counter += 1
    return (a + b) / 2.0


def gsection(func, a, b, a_lst=None, b_lst=None, target="min", epsilon=1e-10, iter_lim=1000000):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    multiplier1, multiplier2 = (3.0 - np.sqrt(5)) / 2.0, (np.sqrt(5) - 1.0) / 2.0
    dot1, dot2 = a + multiplier1 * (b - a), a + multiplier2 * (b - a)
    if a_lst is not None:
        a_lst.append(a)
    if b_lst is not None:
        b_lst.append(b)
    counter = 0
    while b - a > epsilon and counter < iter_lim:
        if sign * func(dot1) > sign * func(dot2):
            a, dot1, dot2 = dot1, dot2, dot1 + multiplier2 * (b - dot1)
        else:
            b, dot1, dot2 = dot2, a + multiplier1 * (dot2 - a), dot1
        if a_lst is not None:
            a_lst.append(a)
        if b_lst is not None:
            b_lst.append(b)
        counter += 1
    return (a + b) / 2.0


def fibonacci(func, a, b, a_lst=None, b_lst=None, target="min", epsilon=1e-10, iter_lim=1000000):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    if a_lst is not None:
        a_lst.append(a)
    if b_lst is not None:
        b_lst.append(b)
    fib_sequence = np.array([1, 1, 2])
    fib_number = 0
    while (b - a) / epsilon > fib_sequence[fib_number + 2] and fib_number < iter_lim:
        fib_sequence = np.append(fib_sequence, fib_sequence[fib_number + 1] + fib_sequence[fib_number + 2])
        fib_number += 1
    fib_sequence = np.array(fib_sequence)
    for i in range(fib_number):
        dot1, dot2 = a + fib_sequence[fib_number - i] / fib_sequence[fib_number - i + 2] * (b - a), \
                     a + fib_sequence[fib_number - i + 1] / fib_sequence[fib_number - i + 2] * (b - a)
        if sign * func(dot1) <= sign * func(dot2):
            b, dot2 = dot2, dot1
        else:
            a, dot1 = dot1, dot2
        if a_lst is not None:
            a_lst.append(a)
        if b_lst is not None:
            b_lst.append(b)
    return (a + b) / 2.0


def tangent(func, a, b, target="min", epsilon=1e-10, iter_lim=1000000, dx=1e-3):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    counter = 0
    while (b - a) / 2.0 > epsilon and counter < iter_lim:
        x = (b * sign * derivative(func, b, dx=dx) - a * sign * derivative(func, a, dx=dx) +
             sign * func(a) - sign * func(b)) / (sign * derivative(func, b, dx=dx) - sign * derivative(func, a, dx=dx))
        if np.abs(derivative(func, x, dx=dx)) < epsilon:
            return x
        elif sign * derivative(func, x, dx=dx) > epsilon:
            b = x
        else:
            a = x
        counter += 1
    return (a + b) / 2.0


def parabolic(func, a, b, target="min", epsilon=1e-10, iter_lim=1000000):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    counter = 0
    x0, x1, x2 = a, (a + b) / 2.0, b
    while sign * func(x1) > np.min([sign * func(x0), sign * func(x2)]) and counter < iter_lim:
        if sign * func(x1) > sign * func(x0):
            x1 = x0 + (b - a) / np.power(2, counter + 2)
            counter += 1
            continue
        if sign * func(x1) > sign * func(x2):
            x1 = x2 - (b - a) / np.power(2, counter + 2)
            counter += 1
            continue
    f0, f1, f2 = sign * func(x0), sign * func(x1), sign * func(x2)
    res = (x0 + x1) / 2.0 + (f1 - f0) * (x2 - x0) * (x2 - x1) / 2 / ((f1 - f0) * (x2 - x0) - (f2 - f0) * (x1 - x0))
    while np.abs(res - x1) >= epsilon and np.abs(res - x2) >= epsilon:
        if res < x1:
            x3, f3, x2, f2, x1, f1 = x2, f2, x1, f1, res, sign * func(res)
        elif res > x1:
            x3, f3, x2, f2 = x2, f2, res, sign * func(res)
        else:
            x3, f3, x2, f2, x1, f1 = x2, f2, x1, f1, (x0 + x1) / 2.0, sign * func((x0 + x1) / 2.0)
        if f1 > f2:
            x0, f0, x1, f1, x2, f2 = x1, f1, x2, f2, x3, f3
        res = (x0 + x1) / 2.0 + (f1 - f0) * (x2 - x0) * (x2 - x1) / 2 / ((f1 - f0) * (x2 - x0) - (f2 - f0) * (x1 - x0))
    return res


def gauss(func, x0, alpha_a, alpha_b, target="min", epsilon=1e-10, iter_lim=1000000):
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    if not (isinstance(x0, np.ndarray) or isinstance(x0, list) or isinstance(x0, float)):
        raise TypeError("x0 had to be an array")
    x0 = np.array(x0)
    x_res, counter, tmp_x1, tmp_x2 = x0.copy(), 0, x0.copy(), np.empty_like(x0)
    results = [x0.copy()]
    while linalg.norm(tmp_x1 - tmp_x2) > epsilon and counter < iter_lim:
        tmp_x1 = x_res.copy()
        for e in np.eye(np.size(x0)):
            alpha = gsection(lambda alpha: sign * func(x_res + alpha * e), alpha_a, alpha_b, iter_lim=iter_lim)
            x_res += alpha * e
        tmp_x2 = x_res.copy()
        results.append(x_res.copy())
        counter += 1
    return np.array(results)


def pattern(func, x0, alpha_a, alpha_b, target="min", epsilon=1e-10, iter_lim=1000000):
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    if not (isinstance(x0, np.ndarray) or isinstance(x0, list) or isinstance(x0, float)):
        raise TypeError("x0 had to be an array")
    x0 = np.array(x0)
    x_res, counter, tmp_x1, tmp_x2 = x0.copy(), 0, x0.copy(), np.empty_like(x0)
    results = [x0.copy()]
    while counter < iter_lim:
        tmp_x1 = x_res.copy()
        for e in np.eye(np.size(x0)):
            alpha = gsection(lambda alpha: sign * func(x_res + alpha * e), alpha_a, alpha_b, iter_lim=iter_lim)
            x_res += alpha * e
        tmp_x2 = x_res.copy()
        results.append(x_res.copy())
        if linalg.norm(tmp_x2 - tmp_x1) < epsilon:
            break
        alpha = gsection(lambda alpha: sign * func(x_res + alpha * (tmp_x2 - tmp_x1)), alpha_a, alpha_b, iter_lim=iter_lim)
        x_res += alpha * (tmp_x2.copy() - tmp_x1.copy())
        results.append(x_res.copy())
        counter += 1
    return np.array(results)


def gradient_step_reduction(func, x0, default_step=10, step_red_mult=0.5, grad_epsilon=1e-8, target="min",
                            epsilon=1e-10, iter_lim=1000000):
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")
    if not (isinstance(x0, np.ndarray) or isinstance(x0, list) or isinstance(x0, float)):
        raise TypeError("x0 had to be an array")
    x0 = np.array(x0)
    counter = 0
    x_current, x_next = x0.copy(), x0.copy()
    results = [x0.copy()]
    # print("alphas")
    while counter < iter_lim:
        x_current = x_next.copy()
        step = default_step
        while func(x_current - step * middle_grad(x_current, func, epsilon=grad_epsilon)) >= func(x_current):
            step *= step_red_mult
        # print("%.6f" % step)
        x_next = x_current - step * middle_grad(x_current, func, epsilon=grad_epsilon)
        results.append(x_next.copy())
        if np.abs(func(x_next) - func(x_current)) < epsilon:
            break
        counter += 1
    return np.array(results)


# Несколько последующих функций реализуют вычисление градиентов по разным схемам
# Эти функции принимают одинаковые параметры и различаются исключительно способом приближенного вычисление градиента
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# x0 - точка, в которой необходимо вычислить градиент
# func - функция, градиент которой нужно найти
# epsilon - точность вычисления градиента
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: приближенное значение градиента функции func в точке x0
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Вспомогательная функция, вычисляющая градиент, как вектор, близкий к вектору левосторонних частных производных
# !!!ВНИМАНИЕ!!! Данную функцию для вычисления градиента возможно использовать только в том случае,
# когда функция func допускает матричный аргумент, причем, при передачи в func матрицы, порядка n x m,
# функция вернет m-мерный вектор, i-я компонента которого будет соответствовать результату примененения функции
# к i-му столбцу
def left_side_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1)) -
            func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) - epsilon * np.eye(x0.size))) / \
           epsilon


# Вспомогательная функция, вычисляющая градиент, как вектор, близкий к вектору правосторонних частных производных
# !!!ВНИМАНИЕ!!! Данную функцию для вычисления градиента возможно использовать только в том случае,
# когда функция func допускает матричный аргумент, причем, при передачи в func матрицы, порядка n x m,
# функция вернет m-мерный вектор, i-я компонента которого будет соответствовать результату примененения функции
# к i-му столбцу
def right_side_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) + epsilon * np.eye(x0.size)) -
            func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1))) / epsilon


# Вспомогательная функция, вычисляющая градиент, как вектор, близкий к вектору среднеарифметических значений
# между правосторонними и левосторонними частными производными
# !!!ВНИМАНИЕ!!! Данную функцию для вычисления градиента возможно использовать только в том случае,
# когда функция func допускает матричный аргумент, причем, при передачи в func матрицы, порядка n x m,
# функция вернет m-мерный вектор, i-я компонента которого будет соответствовать результату примененения функции
# к i-му столбцу
def middle_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) + epsilon * np.eye(x0.size)) -
            func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) - epsilon * np.eye(x0.size))) / \
           2 / epsilon


# Вспомогательная функция, вычисляющая градиент, как вектор, близкий к вектору левосторонних частных производных
# Данная функция медленнее, чем её аналог выше, но её можно применять для всех функций
def left_side_grad_non_matrix(x0, func, epsilon=1e-6):
    gradient, unit_m = np.zeros_like(x0), np.eye(x0.size, x0.size)
    for i in range(x0.size):
        gradient[i] = (func(x0) - func(x0 - epsilon * unit_m[i])) / epsilon
    return gradient


# Вспомогательная функция, вычисляющая градиент, как вектор, близкий к вектору правосторонних частных производных
# Данная функция медленнее, чем её аналог выше, но её можно применять для всех функций
def right_side_grad_non_matrix(x0, func, epsilon=1e-6):
    gradient, unit_m = np.zeros_like(x0), np.eye(x0.size, x0.size)
    for i in range(x0.size):
        gradient[i] = (func(x0 + epsilon * unit_m[i]) - func(x0)) / epsilon
    return gradient


# Вспомогательная функция, вычисляющая градиент, как вектор, близкий к вектору среднеарифметических значений
# между правосторонними и левосторонними частными производными
# Данная функция медленнее, чем её аналог выше, но её можно применять для всех функций
def middle_grad_non_matrix(x0, func, epsilon=1e-6):
    gradient = np.zeros_like(x0)
    unit_m = np.eye(x0.size, x0.size)
    for i in range(x0.size):
        gradient[i] = (func(x0 + epsilon * unit_m[i]) - func(x0 - epsilon * unit_m[i])) / 2 / epsilon
    return gradient


# Вспомогательная функция, вычисляющая градиент, как вектор, близкий к вектору среднеарифметических значений
# между правосторонними и левосторонними частными производными
# Распараллеленная версия функции middle_grad_non_matrix
def middle_grad_non_matrix_pool(x0, func, epsilon=1e-6):
    pool = ThreadPool(np.minimum(x0.size, cpu_count()))
    args_lst = [(i, x0, func, epsilon) for i in range(x0.size)]
    gradient = pool.map(partial_derivative, args_lst)
    pool.close()
    pool.join()
    return np.array(gradient)


def partial_derivative(args):
    i, x0, func, epsilon = args
    unit_m = np.eye(x0.size, x0.size)
    return (func(x0 + epsilon * unit_m[i]) - func(x0 - epsilon * unit_m[i])) / 2 / epsilon


def reduced_gradient_Wolfe(func, x0, A, grad=middle_grad, grad_epsilon=1e-8, target="min", calc_epsilon=1e-10,
                           iter_lim=1000000, infty=1e+5):
    sign = target_input(target)
    x_current = x0_input(x0)
    results = [x_current.copy()]
    f = lambda x: sign * func(x)
    local_infty = float(infty)
    for k in range(iter_lim):
        direction = np.zeros_like(x_current)
        basis_inds, non_basis_inds = \
            np.argpartition(x_current, -A.shape[0])[-A.shape[0]:], \
            np.argpartition(-x_current, -A.shape[1] + A.shape[0])[-A.shape[1] + A.shape[0]:]
        basis, non_basis = A[:, basis_inds], A[:, non_basis_inds]
        difference_vector = grad(x_current, f, grad_epsilon) - \
                            np.dot(np.dot(grad(x_current, f, grad_epsilon)[basis_inds], np.linalg.inv(basis)), A)
        direction[non_basis_inds] = np.where(difference_vector[non_basis_inds] < calc_epsilon,
                                             -difference_vector[non_basis_inds],
                                             -x_current[non_basis_inds] * difference_vector[non_basis_inds])
        direction[basis_inds] = -np.dot(np.dot(np.linalg.inv(basis), non_basis), direction[non_basis_inds])
        # print('%.4f, %.4f, %.4f, %.4f, %.4f' %
        #       (direction[0], direction[1], direction[2], direction[3], direction[4]))
        # print('%.5f' % linalg.norm(direction))
        if linalg.norm(direction) < calc_epsilon:
            break
        max_step = local_infty if np.all(direction > -calc_epsilon)\
            else np.min(-x_current[direction < -calc_epsilon] / direction[direction < -calc_epsilon])
        step = gsection(lambda alpha: f(x_current + alpha * direction), 0.0, max_step, target='min',
                        epsilon=calc_epsilon, iter_lim=iter_lim)
        # print('%.5f' % step)
        x_current = x_current + step * direction
        results.append(x_current.copy())
    return np.array(results)


# Реализация поиска шагового множителя методом скорейшего спуска
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# kwargs - словарь, содержащий параметры данного метода; параметры описаны ниже
# ----------------------------------------------------------------------------------------------------------------------
# Параметры, передаваемые внутри r-алгоритма:
# func - целевая функция
# x_current - текущее приближение к решению
# direction - направление спуска
# ----------------------------------------------------------------------------------------------------------------------
# Параметры передаваемые пользователем:
# step_min - минимальное значение шагового множителя (Идеальный вариант step_min=0.0)
# step_max - максимальное значение шагового множителя (Идеальный вариант step_max=бесконечность)
# argmin_finder - функция, реализующая метод одномерного поиска и имеющая интерфейс функции gsection
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: шаговый множитель, вычисленный методом скорейшего спуска
def step_argmin(kwargs):
    func, x_current, direction, step_min, step_max, argmin_finder = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('step_min'), \
        kwargs.get('step_max'), kwargs.get('argmin_finder')
    return argmin_finder(lambda step: func(x_current - step * direction), step_min, step_max)


# Реализация поиска шагового множителя методом априорного определения
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# kwargs - словарь, содержащий параметры данного метода; параметры описаны ниже
# ----------------------------------------------------------------------------------------------------------------------
# Параметры передаваемые пользователем:
# step_defining_func - функция, явно определяющая шаговый множитель
# step_index - индекс текущей итерации
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: шаговый множитель, вычисленный методом априорного определения
def step_func(kwargs):
    step_defining_func, step_index = kwargs.get('step_defining_func'), kwargs.get('step_index')
    return step_defining_func(step_index)


# Реализация поиска шагового множителя методом дробления шага
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# kwargs - словарь, содержащий параметры данного метода; параметры описаны ниже
# ----------------------------------------------------------------------------------------------------------------------
# Параметры, передаваемые внутри r-алгоритма:
# func - целевая функция
# x_current - текущее приближение к решению
# direction - направление спуска
# ----------------------------------------------------------------------------------------------------------------------
# Параметры передаваемые пользователем:
# default_step - начальный (пробный) шаг
# step_red_mult - коэффициент дробления шага
# reduction_epsilon - число, заменяющее ноль в проверке критерия остановки
# цель параметра - избежать погрешностей в случае равенства функции в старой и новой точках
# step_epsilon - величина, больше которой должен быть шаг
# если же шаг меньше данного числа, то считаем, что шаг приблизительно равен 0.0
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: шаговый множитель, вычисленный методом дробления шага
def step_reduction(kwargs):
    func, x_current, direction, default_step, step_red_mult, reduction_epsilon, step_epsilon = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('default_step'), \
        kwargs.get('step_red_mult'), kwargs.get('reduction_epsilon'), kwargs.get('step_epsilon')
    step = default_step
    while reduction_epsilon >= func(x_current) - func(x_current - step * direction) and np.abs(step) > step_epsilon:
        step *= step_red_mult
    return step


# Реализация поиска шагового множителя при помощи адаптивного алгоритма
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# kwargs - словарь, содержащий параметры данного метода; параметры описаны ниже
# ----------------------------------------------------------------------------------------------------------------------
# Параметры, передаваемые внутри r-алгоритма:
# func - целевая функция
# x_current - текущее приближение к решению
# direction - направление спуска
# ----------------------------------------------------------------------------------------------------------------------
# Параметры передаваемые пользователем:
# default_step - начальный (пробный) шаг
# step_red_mult - коэффициент уменьшения шага
# step_incr_mult - коэффициент увеличения шага
# lim_num - предельное количество шагов, сделанных в одном направлении без увеличения шагового множителя
# reduction_epsilon - число, заменяющее ноль в проверке критерия остановки
# цель параметра - избежать погрешностей в случае равенства функции в старой и новой точках
# step_epsilon - величина, больше которой должен быть шаг
# если же шаг меньше данного числа, то считаем, что шаг приблизительно равен 0.0
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: шаговый множитель, вычисленный при помощи адаптивного алгоритма
def step_adaptive(kwargs):
    func, x_current, direction, default_step, step_red_mult, step_incr_mult, lim_num, reduction_epsilon, step_epsilon,\
        grad, grad_epsilon = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('default_step'), \
        kwargs.get('step_red_mult'), kwargs.get('step_incr_mult'), kwargs.get('lim_num'), \
        kwargs.get('reduction_epsilon'), kwargs.get('step_epsilon'), kwargs.get('grad'), kwargs.get('grad_epsilon')
    step = default_step
    while reduction_epsilon >= func(x_current) - func(x_current - step * direction) and np.abs(step) > step_epsilon:
        step *= step_red_mult
    break_flag = 0
    tmp_step, step = step, 0.0
    while True:
        for i in range(1, lim_num + 1):
            f_old, f_new = \
                func(x_current - (step + (i - 1) * tmp_step) * direction),\
                func(x_current - (step + i * tmp_step) * direction)
            if reduction_epsilon >= f_old - f_new \
                    or isnan(f_old)\
                    or isnan(f_new):
                step += (i - 1) * tmp_step
                break_flag = 1 if i != 1 else 2
                break
        if break_flag == 1 or break_flag == 2:
            break
        step += lim_num * tmp_step
        tmp_step *= step_incr_mult
        x_next = x_current - step * direction
        grad_next = grad(x_next, func, grad_epsilon)
        if np.dot(x_next - x_current, grad_next) > 0:
            break
    if break_flag == 2:
        tmp_step /= step_incr_mult
    return step, tmp_step


def step_adaptive_alternative(kwargs):
    func, x_current, direction, default_step, step_red_mult, step_incr_mult, lim_num, step_lim = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('default_step'), \
        kwargs.get('step_red_mult'), kwargs.get('step_incr_mult'), kwargs.get('lim_num'), kwargs.get('step_lim')
    step = default_step
    i = 0
    func_current, func_next = func(x_current),  func(x_current - step * direction)
    while func_current > func_next and i < step_lim:
        func_current, func_next = func_next, func(x_current - (i + 2) * step * direction)
        i += 1
    if i == 0:
        return step, step * step_red_mult
    if i > lim_num:
        return i * step, step_incr_mult * i * step / lim_num
    return i * step, default_step


# Преобразование матрицы B, посредством умножения на матрицу оператора растяжения пространства
# Преобразование вынесено в отдельную функцию для удобства
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# matrix_B - матрица B
# grad_current - текущий субградиент
# grad_next - следующий субградиент
# beta - коэффициент растяжения пространства
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: преобразованная матрица B
def matrix_B_transformation(matrix_B, grad_current, grad_next, beta):
    r_vector = np.dot(matrix_B.T, grad_next - grad_current)
    r_vector = r_vector / linalg.norm(r_vector)
    return np.dot(matrix_B, np.eye(matrix_B.shape[0], matrix_B.shape[1]) +
                  (beta - 1) * np.dot(r_vector.reshape(r_vector.size, 1), r_vector.reshape(1, r_vector.size)))


# Реализация r-алгоритма в B-форме
# Данная функция осуществляет минимизацию заданной целевой функции при помощи r-алгоритма в B-форме
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# func - целевая функция
# x0 - начальное приближение
# grad - метод вычисления субградиента, интерфейс которого совпадает с интерфейсом методов вычисления градиента выше
# beta - коэффициент растяжения пространства
# step_method - способ нахождения шагового множителя, задающийся в виде строки; допускается 4 значения:
# 1) argmin - метод наискорейшего спуска
# 2) func - метод априорного определения
# 3) reduction - метод дробления шага
# 4) adaptive - адаптивный алгоритм
# step_method_kwargs - словарь содержащий параметры указанного метода нахождения шага;
# содержимое словаря перечислено выше и для каждого метода оно различно
# grad_epsilon - точность вычисления субградиента
# calc_epsilon - параметр, используемый при проверки критерия остановки
# step_epsilon -  если шаг меньше данной величины, то считаем, что шаг приблизительно равен 0.0
# iter_lim - предельное количество итераций
# return_grads - переменная, определяющая возвращать коллекцию всех субградиентов или нет (Детали ниже)
# tqdm_fl - переменная-флаг для печати или не печати прогресс-бара
# continue_transformation - переменная-флаг, определяющая, стоит ли продолжать преобразование пространства в направлении
# разности последних отличающихся субградиентов
# print_iter_index - переменная-флаг, определяющая печатать индекс итерации или нет (Если ==True, то печатать)
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение:
# 1) если return_grads = True, то возвращаемое значение - tuple, первый элемент которой - list всех точек приближения,
# второй - list всех субградиентов в соответствующих точках
# 2) если return_grads = False, то возвращаемое значение - list всех точек приближения
def r_algorithm_B_form(func, x0, grad, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                       calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl, continue_transformation,
                       print_iter_index):
    x_current, x_next, matrix_B, grad_current, grad_next = \
        x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
        np.random.rand(x0.size), grad(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']
    step_method_kwargs['func'] = func
    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad'] = grad
    step_method_kwargs['grad_epsilon'] = grad_epsilon
    results = [x_next.copy()]
    grads = [grad_next.copy()]
    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)
    for k in iterations:
        if print_iter_index:
            print(k)
        xi_current = np.dot(matrix_B.T, grad_next)
        xi_current = xi_current / linalg.norm(xi_current)
        step_method_kwargs['x_current'] = x_next
        step_method_kwargs['direction'] = np.dot(matrix_B, xi_current)
        step_method_kwargs['step_index'] = k
        step_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)
        if isinstance(step_current, tuple):
            step_current, step_method_kwargs['default_step'] = step_current
        if np.abs(step_current) < step_epsilon and step_method in continuing_step_methods and continue_transformation:
            matrix_B = matrix_B_transformation(matrix_B, grad_current, grad_next, beta)
            continue
        x_current, grad_current = x_next.copy(), grad_next.copy()
        x_next = x_current - step_current * np.dot(matrix_B, xi_current)
        # print(x_next)
        # print(func(x_next))
        results.append(x_next.copy())
        grad_next = grad(x_next, func, epsilon=grad_epsilon)
        grads.append(grad_next.copy())
        if linalg.norm(x_next - x_current) < calc_epsilon_x or linalg.norm(grad_next) < calc_epsilon_grad:
            break
        matrix_B = matrix_B_transformation(matrix_B, grad_current, grad_next, beta)
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_B_form_double(func_1, func_2, x0_1, x0_2, grad_1, grad_2, beta, step_method, step_method_kwargs,
                              grad_epsilon, calc_epsilon_x, calc_epsilon_grad, step_epsilon, iter_lim,
                              return_grads, tqdm_fl, continue_transformation, print_iter_index):

    x_1_current, x_1_next, matrix_B_1, grad_1_current, grad_1_next = \
        x0_1.copy(), x0_1.copy(), np.eye(x0_1.size, x0_1.size), np.random.rand(x0_1.size),\
        grad_1(x0_1, func_1, epsilon=grad_epsilon)
    x_2_current, x_2_next, matrix_B_2, grad_2_current, grad_2_next = \
        x0_2.copy(), x0_2.copy(), np.eye(x0_2.size, x0_2.size), np.random.rand(x0_2.size),\
        grad_2(x0_2, func_2, epsilon=grad_epsilon)

    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']

    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad_epsilon'] = grad_epsilon

    results_1 = [x_1_next.copy()]
    grads_1 = [grad_1_next.copy()]

    results_2 = [x_2_next.copy()]
    grads_2 = [grad_2_next.copy()]

    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)

    if 'default_step' in step_method_kwargs:
        default_step_1, default_step_2 = step_method_kwargs['default_step'], step_method_kwargs['default_step']

    for k in iterations:

        if print_iter_index:
            print(k)

        xi_1_current = np.dot(matrix_B_1.T, grad_1_next)
        xi_1_current = xi_1_current / linalg.norm(xi_1_current)

        xi_2_current = np.dot(matrix_B_2.T, grad_2_next)
        xi_2_current = xi_2_current / linalg.norm(xi_2_current)

        step_method_kwargs['func'] = func_1
        step_method_kwargs['grad'] = grad_1
        step_method_kwargs['x_current'] = x_1_next
        step_method_kwargs['direction'] = np.dot(matrix_B_1, xi_1_current)
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_1
        step_1_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        step_method_kwargs['func'] = func_2
        step_method_kwargs['grad'] = grad_2
        step_method_kwargs['x_current'] = x_2_next
        step_method_kwargs['direction'] = np.dot(matrix_B_2, xi_2_current)
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_2
        step_2_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        if isinstance(step_1_current, tuple):
            step_1_current, default_step_1 = step_1_current

        if isinstance(step_2_current, tuple):
            step_2_current, default_step_2 = step_2_current

        if (np.abs(step_1_current) < step_epsilon or np.abs(step_2_current) < step_epsilon) and \
                step_method in continuing_step_methods and continue_transformation:

            matrix_B_1 = matrix_B_transformation(matrix_B_1, grad_1_current, grad_1_next, beta)
            matrix_B_2 = matrix_B_transformation(matrix_B_2, grad_2_current, grad_2_next, beta)
            continue

        x_1_current, grad_1_current = x_1_next.copy(), grad_1_next.copy()
        x_1_next = x_1_current - step_1_current * np.dot(matrix_B_1, xi_1_current)
        results_1.append(x_1_next.copy())
        grad_1_next = grad_1(x_1_next, func_1, epsilon=grad_epsilon)
        grads_1.append(grad_1_next.copy())

        x_2_current, grad_2_current = x_2_next.copy(), grad_2_next.copy()
        x_2_next = x_2_current - step_2_current * np.dot(matrix_B_2, xi_2_current)
        results_2.append(x_2_next.copy())
        grad_2_next = grad_2(x_2_next, func_2, epsilon=grad_epsilon)
        grads_2.append(grad_2_next.copy())

        if linalg.norm(np.concatenate((x_1_next, x_2_next)) -
                       np.concatenate((x_1_current, x_2_current))) < calc_epsilon_x or \
           linalg.norm(np.concatenate((grad_1_next, grad_2_next))) < calc_epsilon_grad:
            break

        matrix_B_1 = matrix_B_transformation(matrix_B_1, grad_1_current, grad_1_next, beta)
        matrix_B_2 = matrix_B_transformation(matrix_B_2, grad_2_current, grad_2_next, beta)

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


def r_algorithm_B_form_cooperative(func_1, func_2, x0_1, x0_2, grad_1, grad_2, beta, step_method, step_method_kwargs,
                                   grad_epsilon, calc_epsilon_x, calc_epsilon_grad, step_epsilon, iter_lim,
                                   return_grads, tqdm_fl, continue_transformation, print_iter_index):

    x_1_current, x_1_next, matrix_B_1, grad_1_current, grad_1_next = \
        x0_1.copy(), x0_1.copy(), np.eye(x0_1.size, x0_1.size), np.random.rand(x0_1.size),\
        grad_1(x0_1, lambda x: func_1(x, x0_2), epsilon=grad_epsilon)
    x_2_current, x_2_next, matrix_B_2, grad_2_current, grad_2_next = \
        x0_2.copy(), x0_2.copy(), np.eye(x0_2.size, x0_2.size), np.random.rand(x0_2.size),\
        grad_2(x0_2, lambda x: func_2(x0_1, x), epsilon=grad_epsilon)

    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']

    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad_epsilon'] = grad_epsilon

    results_1 = [x_1_next.copy()]
    grads_1 = [grad_1_next.copy()]

    results_2 = [x_2_next.copy()]
    grads_2 = [grad_2_next.copy()]

    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)

    if 'default_step' in step_method_kwargs:
        default_step_1, default_step_2 = step_method_kwargs['default_step'], step_method_kwargs['default_step']

    for k in iterations:

        step_1_current_zero, step_2_current_zero = False, False

        if print_iter_index:
            print(k)

        xi_1_current = np.dot(matrix_B_1.T, grad_1_next)
        xi_1_current = xi_1_current / linalg.norm(xi_1_current)

        xi_2_current = np.dot(matrix_B_2.T, grad_2_next)
        xi_2_current = xi_2_current / linalg.norm(xi_2_current)

        step_method_kwargs['func'] = lambda x: func_1(x, x_2_next)
        step_method_kwargs['grad'] = grad_1
        step_method_kwargs['x_current'] = x_1_next
        step_method_kwargs['direction'] = np.dot(matrix_B_1, xi_1_current)
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_1
        step_1_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        step_method_kwargs['func'] = lambda x: func_2(x_1_next, x)
        step_method_kwargs['grad'] = grad_2
        step_method_kwargs['x_current'] = x_2_next
        step_method_kwargs['direction'] = np.dot(matrix_B_2, xi_2_current)
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_2
        step_2_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        if isinstance(step_1_current, tuple):
            step_1_current, default_step_1 = step_1_current

        if isinstance(step_2_current, tuple):
            step_2_current, default_step_2 = step_2_current

        if (np.abs(step_1_current) < step_epsilon or np.abs(step_2_current) < step_epsilon) and \
                step_method in continuing_step_methods and continue_transformation:

            matrix_B_1 = matrix_B_transformation(matrix_B_1, grad_1_current, grad_1_next, beta)
            matrix_B_2 = matrix_B_transformation(matrix_B_2, grad_2_current, grad_2_next, beta)
            continue

        if np.abs(step_1_current) < 1e-51:
            step_1_current_zero = True
        else:
            x_1_current, grad_1_current = x_1_next.copy(), grad_1_next.copy()
            x_1_next = x_1_current - step_1_current * np.dot(matrix_B_1, xi_1_current)
        results_1.append(x_1_next.copy())


        if np.abs(step_2_current) < 1e-51:
            step_2_current_zero = True
        else:
            x_2_current, grad_2_current = x_2_next.copy(), grad_2_next.copy()
            x_2_next = x_2_current - step_2_current * np.dot(matrix_B_2, xi_2_current)
        results_2.append(x_2_next.copy())

        grad_1_next = grad_1(x_1_next, lambda x: func_1(x, x_2_next), epsilon=grad_epsilon)
        grads_1.append(grad_1_next.copy())

        grad_2_next = grad_2(x_2_next, lambda x: func_2(x_1_next, x), epsilon=grad_epsilon)
        grads_2.append(grad_2_next.copy())

        if linalg.norm(np.concatenate((x_1_next, x_2_next)) -
                       np.concatenate((x_1_current, x_2_current))) < calc_epsilon_x or \
           linalg.norm(np.concatenate((grad_1_next, grad_2_next))) < calc_epsilon_grad or \
                (step_1_current_zero and step_2_current_zero):
            break
        matrix_B_1 = matrix_B_transformation(matrix_B_1, grad_1_current, grad_1_next, beta)
        matrix_B_2 = matrix_B_transformation(matrix_B_2, grad_2_current, grad_2_next, beta)

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


# Преобразование матрицы H
# Преобразование вынесено в отдельную функцию для удобства
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# matrix_H - матрица H
# grad_current - текущий субградиент
# grad_next - следующий субградиент
# beta - коэффициент растяжения пространства
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: преобразованная матрица H
def matrix_H_transformation(matrix_H, grad_current, grad_next, beta):
    r_vector = grad_next - grad_current
    return matrix_H + (beta * beta - 1) * np.dot(np.dot(matrix_H, r_vector).reshape(r_vector.size, 1),
                                                 np.dot(matrix_H, r_vector).reshape(1, r_vector.size)) / \
           np.dot(np.dot(r_vector, matrix_H), r_vector)


# Реализация r-алгоритма в H-форме
# Данная функция осуществляет минимизацию заданной целевой функции при помощи r-алгоритма в H-форме
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# func - целевая функция
# x0 - начальное приближение
# grad - метод вычисления субградиента, интерфейс которого совпадает с интерфейсом методов вычисления градиента выше
# beta - коэффициент растяжения пространства
# step_method - способ нахождения шагового множителя, задающийся в виде строки; допускается 4 значения:
# 1) argmin - метод наискорейшего спуска
# 2) func - метод априорного определения
# 3) reduction - метод дробления шага
# 4) adaptive - адаптивный алгоритм
# step_method_kwargs - словарь содержащий параметры указанного метода нахождения шага;
# содержимое словаря перечислено выше и для каждого метода оно различно
# grad_epsilon - точность вычисления субградиента
# calc_epsilon - параметр, используемый при проверки критерия остановки
# step_epsilon -  если шаг меньше данной величины, то считаем, что шаг приблизительно равен 0.0
# iter_lim - предельное количество итераций
# return_grads - переменная, определяющая возвращать коллекцию всех субградиентов или нет (Детали ниже)
# tqdm_fl - переменная-флаг для печати или не печати прогресс-бара
# continue_transformation - переменная-флаг, определяющая, стоит ли продолжать преобразование пространства в направлении
# разности последних отличающихся субградиентов
# print_iter_index - переменная-флаг, определяющая печатать индекс итерации или нет (Если ==True, то печатать)
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение:
# 1) если return_grads = True, то возвращаемое значение - tuple, первый элемент которой - list всех точек приближения,
# второй - list всех субградиентов в соответствующих точках
# 2) если return_grads = False, то возвращаемое значение - list всех точек приближения
def r_algorithm_H_form(func, x0, grad, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                       calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl, continue_transformation,
                       print_iter_index):
    x_current, x_next, matrix_H, grad_current, grad_next = \
        x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
        np.random.rand(x0.size), grad(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']
    step_method_kwargs['func'] = func
    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad'] = grad
    step_method_kwargs['grad_epsilon'] = grad_epsilon
    results = [x_next.copy()]
    grads = [grad_next.copy()]
    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)
    for k in iterations:
        if print_iter_index:
            print(k)
        step_method_kwargs['x_current'] = x_next
        step_method_kwargs['direction'] = np.dot(matrix_H, grad_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H, grad_next), grad_next))
        step_method_kwargs['step_index'] = k
        step_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)
        if isinstance(step_current, tuple):
            step_current, step_method_kwargs['default_step'] = step_current
        if np.abs(step_current) < step_epsilon and step_method in continuing_step_methods and continue_transformation:
            matrix_H = matrix_H_transformation(matrix_H, grad_current, grad_next, beta)
            continue
        x_current, grad_current = x_next.copy(), grad_next.copy()
        x_next = x_current - step_current * np.dot(matrix_H, grad_current) / \
                             np.sqrt(np.dot(np.dot(matrix_H, grad_current), grad_current))
        results.append(x_next.copy())
        grad_next = grad(x_next, func, epsilon=grad_epsilon)
        grads.append(grad_next.copy())
        if linalg.norm(x_next - x_current) < calc_epsilon_x or linalg.norm(grad_next) < calc_epsilon_grad:
            break
        matrix_H = matrix_H_transformation(matrix_H, grad_current, grad_next, beta)
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_H_form_double(func_1, func_2, x0_1, x0_2, grad_1, grad_2, beta, step_method, step_method_kwargs,
                              grad_epsilon, calc_epsilon_x, calc_epsilon_grad, step_epsilon, iter_lim,
                              return_grads, tqdm_fl, continue_transformation, print_iter_index):

    x_1_current, x_1_next, matrix_H_1, grad_1_current, grad_1_next = \
        x0_1.copy(), x0_1.copy(), np.eye(x0_1.size, x0_1.size), np.random.rand(x0_1.size),\
        grad_1(x0_1, func_1, epsilon=grad_epsilon)
    x_2_current, x_2_next, matrix_H_2, grad_2_current, grad_2_next = \
        x0_2.copy(), x0_2.copy(), np.eye(x0_2.size, x0_2.size), np.random.rand(x0_2.size),\
        grad_2(x0_2, func_2, epsilon=grad_epsilon)

    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']

    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad_epsilon'] = grad_epsilon

    results_1 = [x_1_next.copy()]
    grads_1 = [grad_1_next.copy()]

    results_2 = [x_2_next.copy()]
    grads_2 = [grad_2_next.copy()]

    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)

    if 'default_step' in step_method_kwargs:
        default_step_1, default_step_2 = step_method_kwargs['default_step'], step_method_kwargs['default_step']

    for k in iterations:

        if print_iter_index:
            print(k)

        step_method_kwargs['func'] = func_1
        step_method_kwargs['grad'] = grad_1
        step_method_kwargs['x_current'] = x_1_next
        step_method_kwargs['direction'] = np.dot(matrix_H_1, grad_1_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H_1, grad_1_next), grad_1_next))
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_1
        step_1_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        step_method_kwargs['func'] = func_2
        step_method_kwargs['grad'] = grad_2
        step_method_kwargs['x_current'] = x_2_next
        step_method_kwargs['direction'] = np.dot(matrix_H_2, grad_2_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H_2, grad_2_next), grad_2_next))
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_2
        step_2_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        if isinstance(step_1_current, tuple):
            step_1_current, default_step_1 = step_1_current

        if isinstance(step_2_current, tuple):
            step_2_current, default_step_2 = step_2_current

        if (np.abs(step_1_current) < step_epsilon or np.abs(step_2_current) < step_epsilon) and \
                step_method in continuing_step_methods and continue_transformation:

            matrix_H_1 = matrix_H_transformation(matrix_H_1, grad_1_current, grad_1_next, beta)
            matrix_H_2 = matrix_H_transformation(matrix_H_2, grad_2_current, grad_2_next, beta)
            continue

        x_1_current, grad_1_current = x_1_next.copy(), grad_1_next.copy()
        x_1_next = x_1_current - step_1_current * np.dot(matrix_H_1, grad_1_next) / \
                   np.sqrt(np.dot(np.dot(matrix_H_1, grad_1_next), grad_1_next))
        results_1.append(x_1_next.copy())
        grad_1_next = grad_1(x_1_next, func_1, epsilon=grad_epsilon)
        grads_1.append(grad_1_next.copy())

        x_2_current, grad_2_current = x_2_next.copy(), grad_2_next.copy()
        x_2_next = x_2_current - step_2_current * np.dot(matrix_H_2, grad_2_next) /\
                   np.sqrt(np.dot(np.dot(matrix_H_2, grad_2_next), grad_2_next))
        results_2.append(x_2_next.copy())
        grad_2_next = grad_2(x_2_next, func_2, epsilon=grad_epsilon)
        grads_2.append(grad_2_next.copy())

        if linalg.norm(np.concatenate((x_1_next, x_2_next)) -
                       np.concatenate((x_1_current, x_2_current))) < calc_epsilon_x or \
           linalg.norm(np.concatenate((grad_1_next, grad_2_next))) < calc_epsilon_grad:
            break

        matrix_H_1 = matrix_H_transformation(matrix_H_1, grad_1_current, grad_1_next, beta)
        matrix_H_2 = matrix_H_transformation(matrix_H_2, grad_2_current, grad_2_next, beta)

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


def r_algorithm_H_form_cooperative(func_1, func_2, x0_1, x0_2, grad_1, grad_2, beta, step_method, step_method_kwargs,
                                   grad_epsilon, calc_epsilon_x, calc_epsilon_grad, step_epsilon, iter_lim,
                                   return_grads, tqdm_fl, continue_transformation, print_iter_index):

    x_1_current, x_1_next, matrix_H_1, grad_1_current, grad_1_next = \
        x0_1.copy(), x0_1.copy(), np.eye(x0_1.size, x0_1.size), np.random.rand(x0_1.size),\
        grad_1(x0_1, lambda x: func_1(x, x0_2), epsilon=grad_epsilon)

    x_2_current, x_2_next, matrix_H_2, grad_2_current, grad_2_next = \
        x0_2.copy(), x0_2.copy(), np.eye(x0_2.size, x0_2.size), np.random.rand(x0_2.size),\
        grad_2(x0_2, lambda x: func_2(x0_1, x), epsilon=grad_epsilon)

    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']

    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad_epsilon'] = grad_epsilon

    results_1 = [x_1_next.copy()]
    grads_1 = [grad_1_next.copy()]

    results_2 = [x_2_next.copy()]
    grads_2 = [grad_2_next.copy()]

    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)

    if 'default_step' in step_method_kwargs:
        default_step_1, default_step_2 = step_method_kwargs['default_step'], step_method_kwargs['default_step']

    for k in iterations:

        step_1_current_zero, step_2_current_zero = False, False

        if print_iter_index:
            print(k)
            print(x_1_next)
            print(x_2_next)
            print('Вычисление шага №1')

        step_method_kwargs['func'] = lambda x: func_1(x, x_2_next)
        step_method_kwargs['grad'] = grad_1
        step_method_kwargs['x_current'] = x_1_next
        step_method_kwargs['direction'] = np.dot(matrix_H_1, grad_1_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H_1, grad_1_next), grad_1_next))
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_1
        step_1_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        if print_iter_index:
            print('Вычисление шага №2')

        step_method_kwargs['func'] = lambda x: func_2(x_1_next, x)
        step_method_kwargs['grad'] = grad_2
        step_method_kwargs['x_current'] = x_2_next
        step_method_kwargs['direction'] = np.dot(matrix_H_2, grad_2_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H_2, grad_2_next), grad_2_next))
        step_method_kwargs['step_index'] = k
        if 'default_step' in step_method_kwargs:
            step_method_kwargs['default_step'] = default_step_2
        step_2_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)

        if isinstance(step_1_current, tuple):
            step_1_current, default_step_1 = step_1_current

        if isinstance(step_2_current, tuple):
            step_2_current, default_step_2 = step_2_current

        if (np.abs(step_1_current) < step_epsilon or np.abs(step_2_current) < step_epsilon) and \
                step_method in continuing_step_methods and continue_transformation:

            matrix_H_1 = matrix_H_transformation(matrix_H_1, grad_1_current, grad_1_next, beta)
            matrix_H_2 = matrix_H_transformation(matrix_H_2, grad_2_current, grad_2_next, beta)
            continue

        if print_iter_index:
            print('Вычисление приближения №1')

        if np.abs(step_1_current) < 1e-51:
            step_1_current_zero = True
        else:
            x_1_current, grad_1_current = x_1_next.copy(), grad_1_next.copy()
            x_1_next = x_1_current - step_1_current * np.dot(matrix_H_1, grad_1_next) / \
                       np.sqrt(np.dot(np.dot(matrix_H_1, grad_1_next), grad_1_next))
        results_1.append(x_1_next.copy())

        if print_iter_index:
            print('Вычисление приближения №2')

        if np.abs(step_2_current) < 1e-51:
            step_2_current_zero = True
        else:
            x_2_current, grad_2_current = x_2_next.copy(), grad_2_next.copy()
            x_2_next = x_2_current - step_2_current * np.dot(matrix_H_2, grad_2_next) / \
                       np.sqrt(np.dot(np.dot(matrix_H_2, grad_2_next), grad_2_next))
        results_2.append(x_2_next.copy())

        if print_iter_index:
            print('Вычисление градиента №1')

        grad_1_next = grad_1(x_1_next, lambda x: func_1(x, x_2_next), epsilon=grad_epsilon)
        grads_1.append(grad_1_next.copy())

        if print_iter_index:
            print('Вычисление градиента №2')

        grad_2_next = grad_2(x_2_next, lambda x: func_2(x_1_next, x), epsilon=grad_epsilon)
        grads_2.append(grad_2_next.copy())

        if linalg.norm(np.concatenate((x_1_next, x_2_next)) -
                       np.concatenate((x_1_current, x_2_current))) < calc_epsilon_x or \
           linalg.norm(np.concatenate((grad_1_next, grad_2_next))) < calc_epsilon_grad or \
                (step_1_current_zero and step_2_current_zero):
            break

        if print_iter_index:
            print('Преобразование матриц')

        matrix_H_1 = matrix_H_transformation(matrix_H_1, grad_1_current, grad_1_next, beta)
        matrix_H_2 = matrix_H_transformation(matrix_H_2, grad_2_current, grad_2_next, beta)

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


# Вспомогательный метод, осуществляющий проверку правильно ли введен тип задачи оптимизации (мин. или макс.)
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# target_dual - строковая переменная, отвечающая за тип задачи оптимизации (минимизация или максимизация)
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение:
# 1) если задача на минимум: 1.0
# 2) если задача на максимум: -1.0
# 3) исключение, если target_dual задано в неправильном формате
def target_input(target):
    if target.lower() == "min" or target.lower() == "minimum":
        return 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        return -1.0
    else:
        raise ValueError("invalid value of \"target_dual\"")


# Вспомогательный метод, осуществляющий преобразование начального приближения к типу numpy.ndarray
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# x0 - начальное приближение
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: неглубокая копия x0, преобразованная к типу numpy.ndarray
def x0_input(x0):
    return np.array(x0).copy()


# Функция, принимающая начальные параметры, анализирующая их, и вызывающая указанную версию r-алгоритма
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# func - целевая функция
# x0 - начальное приближение
# args - tuple дополнительных параметров целевой функции, которые на момент оптимизации считаются постоянными
# grad - метод вычисления субградиента, интерфейс которого совпадает с интерфейсом методов вычисления градиента выше
# form - форма r-алгоритма (если в form содержится буква 'H', то алгоритм применяется в H-форме;
# в противном случае: в B-форме)
# beta - коэффициент растяжения пространства
# target_dual - тип задачи оптимизации (макс. или мин.); допустимые значения те же, что и ранее
# grad_epsilon - точность вычисления субградиента
# calc_epsilon - параметр, используемый при проверки критерия остановки
# step_epsilon -  если шаг меньше данной величины, то считаем, что шаг приблизительно равен 0.0
# iter_lim - предельное количество итераций
# return_grads - переменная, определяющая возвращать коллекцию всех субградиентов или нет (Детали ниже)
# tqdm_fl - переменная-флаг для печати или не печати прогресс-бара
# continue_transformation - переменная-флаг, определяющая, стоит ли продолжать преобразование пространства в направлении
# разности последних отличающихся субградиентов
# print_iter_index - переменная-флаг, определяющая печатать индекс итерации или нет (Если ==True, то печатать)
# kwargs - неопределенное количество именованных параметров, которые определят метод нахождения шагового множителя;
# именованные параметры задаются по следующим правилам:
# 1) Для указания способа нахождения шагового множителя, необходимо при вызове данной функции указать
# step_method=[допустимое значение], после чего перечислить параметры переданного метода (параметры для каждого метода
# см. в.);
# для step_method допускаются следующие значения:
#     а) argmin - метод наискорейшего спуска
#     б) func - метод априорного определения
#     в) reduction - метод дробления шага
#     г) adaptive - адаптивный алгоритм
# 2) Если step_method не будет указано, как и не будет указано никаких других дополнительных аргументов, помимо
# перечисленных до kwargs, то будет применен адаптивный алгоритм с некоторыми параметрами по-умолчанию
# 3) Если часть параметров для метода нахождения шагового множителя будет не указана, то будет сгенерировано исключение
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение:
# 1) если return_grads = True, то возвращаемое значение - tuple, первый элемент которой - list всех точек приближения,
# второй - list всех субградиентов в соответствующих точках
# 2) если return_grads = False, то возвращаемое значение - list всех точек приближения
def r_algorithm(func, x0, args=None, grad=middle_grad_non_matrix, form='B', beta=0.5, target='min',
                grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10, step_epsilon=1e-15, iter_lim=1000000,
                return_grads=False, tqdm_fl=False, continue_transformation=True, print_iter_index=False, **kwargs):
    sign = target_input(target)
    x0 = x0_input(x0)
    step_method_kwargs = {}
    if len(kwargs) > 0:
        for key in kwargs.keys():
            step_method_kwargs[key] = kwargs.get(key)
    else:
        step_method_kwargs['step_method'] = 'adaptive'
        step_method_kwargs['default_step'] = 10.0
        step_method_kwargs['step_red_mult'] = 0.5
        step_method_kwargs['step_incr_mult'] = 1.2
        step_method_kwargs['lim_num'] = 3
        step_method_kwargs['reduction_epsilon'] = 1e-15
    step_method_kwargs['step_epsilon'] = step_epsilon
    step_method = step_method_kwargs.get('step_method')
    if args is None:
        func_as_arg = lambda x: sign * func(x)
    else:
        func_as_arg = lambda x: sign * func(x, args)
    if 'H' in form:
        return r_algorithm_H_form(func_as_arg, x0, grad, beta, step_method, step_method_kwargs,
                                  grad_epsilon=grad_epsilon, calc_epsilon_x=calc_epsilon_x,
                                  calc_epsilon_grad=calc_epsilon_grad, step_epsilon=step_epsilon, iter_lim=iter_lim,
                                  return_grads=return_grads, tqdm_fl=tqdm_fl,
                                  continue_transformation=continue_transformation, print_iter_index=print_iter_index)
    else:
        return r_algorithm_B_form(func_as_arg, x0, grad, beta, step_method, step_method_kwargs,
                                  grad_epsilon=grad_epsilon, calc_epsilon_x=calc_epsilon_x,
                                  calc_epsilon_grad=calc_epsilon_grad, step_epsilon=step_epsilon, iter_lim=iter_lim,
                                  return_grads=return_grads, tqdm_fl=tqdm_fl,
                                  continue_transformation=continue_transformation, print_iter_index=print_iter_index)


def r_algorithm_double(func_1, func_2, x0_1, x0_2, args_1=None, args_2=None, grad_1=middle_grad_non_matrix_pool,
                       grad_2=middle_grad_non_matrix_pool, form='B', beta=0.5, target_1='min', target_2='min',
                       grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10, step_epsilon=1e-15,
                       iter_lim=1000000, return_grads=False, tqdm_fl=False, continue_transformation=True,
                       print_iter_index=False, **kwargs):

    sign_1, sign_2 = target_input(target_1), target_input(target_2)
    x0_1, x0_2 = x0_input(x0_1), x0_input(x0_2)

    step_method_kwargs = {}
    if len(kwargs) > 0:
        for key in kwargs.keys():
            step_method_kwargs[key] = kwargs.get(key)
    else:
        step_method_kwargs['step_method'] = 'adaptive'
        step_method_kwargs['default_step'] = 10.0
        step_method_kwargs['step_red_mult'] = 0.5
        step_method_kwargs['step_incr_mult'] = 1.2
        step_method_kwargs['lim_num'] = 3
        step_method_kwargs['reduction_epsilon'] = 1e-15
    step_method_kwargs['step_epsilon'] = step_epsilon
    step_method = step_method_kwargs.get('step_method')

    if args_1 is None:
        func_as_arg_1 = lambda x: sign_1 * func_1(x)
    else:
        func_as_arg_1 = lambda x: sign_1 * func_1(x, args_1)

    if args_2 is None:
        func_as_arg_2 = lambda x: sign_2 * func_2(x)
    else:
        func_as_arg_2 = lambda x: sign_2 * func_2(x, args_2)

    if 'H' in form:
        return r_algorithm_H_form_double(func_as_arg_1, func_as_arg_2, x0_1, x0_2, grad_1, grad_2, beta,
                                         step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                                         calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                         continue_transformation, print_iter_index)
    else:
        return r_algorithm_B_form_double(func_as_arg_1, func_as_arg_2, x0_1, x0_2, grad_1, grad_2, beta,
                                         step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                                         calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                         continue_transformation, print_iter_index)


def r_algorithm_cooperative(func_1, func_2, x0_1, x0_2, args_1=None, args_2=None, grad_1=middle_grad_non_matrix_pool,
                            grad_2=middle_grad_non_matrix_pool, form='B', beta=0.5, target_1='min', target_2='min',
                            grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10, step_epsilon=1e-15,
                            iter_lim=1000000, return_grads=False, tqdm_fl=False, continue_transformation=True,
                            print_iter_index=False, **kwargs):

    sign_1, sign_2 = target_input(target_1), target_input(target_2)
    x0_1, x0_2 = x0_input(x0_1), x0_input(x0_2)

    step_method_kwargs = {}
    if len(kwargs) > 0:
        for key in kwargs.keys():
            step_method_kwargs[key] = kwargs.get(key)
    else:
        step_method_kwargs['step_method'] = 'adaptive'
        step_method_kwargs['default_step'] = 10.0
        step_method_kwargs['step_red_mult'] = 0.5
        step_method_kwargs['step_incr_mult'] = 1.2
        step_method_kwargs['lim_num'] = 3
        step_method_kwargs['reduction_epsilon'] = 1e-15
    step_method_kwargs['step_epsilon'] = step_epsilon
    step_method = step_method_kwargs.get('step_method')

    if args_1 is None:
        func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y)
    else:
        func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y, args_1)

    if args_2 is None:
        func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y)
    else:
        func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y, args_2)

    if 'H' in form:
        return r_algorithm_H_form_cooperative(func_as_arg_1, func_as_arg_2, x0_1, x0_2, grad_1, grad_2, beta,
                                              step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                                              calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                              continue_transformation, print_iter_index)
    else:
        return r_algorithm_B_form_cooperative(func_as_arg_1, func_as_arg_2, x0_1, x0_2, grad_1, grad_2, beta,
                                              step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                                              calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                              continue_transformation, print_iter_index)


# Вспомогательная функция, которая для заданной последовательности точек, возвращает точки, что отличаются от своей
# предыдущей более чем на eps
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# points - список точек
# eps - минимальное расстояние, между точками в новом списке
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: numpy.ndarray точек из points, без точек, что отличаются от своих предыдущих больше, чем на eps
def remove_nearly_same_points(points, eps=1e-3):
    results = [points[0].copy()]
    for i in range(len(points) - 1):
        if np.linalg.norm(results[0] - points[i]) > eps:
            results.insert(0, points[i].copy())
    results.insert(0, points[len(points) - 1])
    return np.array(results[::-1])


def feasible_region_indicator_linear(x, A, b):
    result = np.array(np.ones_like(A[0, 0] * x[0]), dtype=bool)
    for i in range(A.shape[0]):
        sum = 0
        for j in range(A.shape[1]):
            sum += A[i, j] * x[j]
        result = np.logical_and(sum <= b[i], result)
    return np.array(result, dtype=float)


def feasible_region_indicator(x, g):
    result = np.array(np.ones_like(g[0](x)), dtype=bool)
    for i in range(len(g)):
        result = np.logical_and(g[i](x) <= 0, result)
    return np.array(result, dtype=float)


# Метод внутренней точки с штрафными функциями вида -rk * sum(i=1, m, ln(-gi(x)))
def r_algorithm_interior_point_1(func, x0, g, r_k, args=None, grad=middle_grad_non_matrix, form='B', beta=0.5,
                                 target='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10,
                                 step_epsilon=1e-15, r_epsilon=1e-10, iter_lim=1000000, return_grads=False,
                                 tqdm_fl=False, continue_transformation=True, print_iter_index=False, **kwargs):
    sign = target_input(target)
    x_current, x_next = np.random.rand(x0.size), x0_input(x0)
    k = 0
    results, grads = [x_next.copy()], []
    while np.linalg.norm(x_current - x_next) > calc_epsilon_x and r_k(k) > r_epsilon and k < iter_lim:
        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_current = x_next.copy()

        if args is None:
            x_next = r_algorithm(lambda x: sign * func(x) -
                                           r_k(k) * np.array([np.log(-g[i](x)) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)
        else:
            x_next = r_algorithm(lambda x: sign * func(x, args) -
                                           r_k(k) * np.array([np.log(-g[i](x)) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)

        if type(x_next) is tuple:
            grads.append(x_next[1].copy())
            x_next = x_next[0].copy()
        x_next = x_next[-1].copy()
        # if print_iter_index:
        #     print(x_next)
        #     if args is None:
        #         print(func(x_next))
        #     else:
        #         print(func(x_next, args))
        results.append(x_next.copy())
        k += 1
        # if print_iter_index:
        #     print('---------------------------')
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_interior_point_1_cooperative(func_1, func_2, x0_1, x0_2, g_1, g_2, r_k_1, r_k_2, args_1=None,
                                             args_2=None, grad_1=middle_grad_non_matrix_pool,
                                             grad_2=middle_grad_non_matrix_pool, form='B', beta=0.5, target_1='min',
                                             target_2='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10,
                                             calc_epsilon_grad=1e-10, step_epsilon=1e-15, r_epsilon=1e-10,
                                             iter_lim=1000000, return_grads=False, tqdm_fl=False,
                                             continue_transformation=True, print_iter_index=False, **kwargs):

    sign_1, sign_2 = target_input(target_1), target_input(target_2)
    x_1_current, x_1_next = np.random.rand(x0_1.size), x0_input(x0_1)
    x_2_current, x_2_next = np.random.rand(x0_2.size), x0_input(x0_2)
    k = 0
    results_1, grads_1 = [x_1_next.copy()], []
    results_2, grads_2 = [x_2_next.copy()], []

    while np.linalg.norm(np.concatenate((x_1_current, x_2_current)) - np.concatenate((x_1_next, x_2_next))) > \
            calc_epsilon_x and r_k_1(k) > r_epsilon and r_k_2(k) > r_epsilon and k < iter_lim:

        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_1_current, x_2_current = x_1_next.copy(), x_2_next.copy()

        if args_1 is None:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y) - r_k_1(k) * \
                                         np.array([np.log(-g_1[i](x, y)) for i in range(len(g_1))]).sum()
        else:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y, args_1) - r_k_1(k) * \
                                         np.array([np.log(-g_1[i](x, y)) for i in range(len(g_1))]).sum()

        if args_2 is None:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y) - r_k_2(k) * \
                                         np.array([np.log(-g_2[i](x, y)) for i in range(len(g_2))]).sum()
        else:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y, args_2) - r_k_2(k) * \
                                         np.array([np.log(-g_2[i](x, y)) for i in range(len(g_2))]).sum()

        r_alg_result = r_algorithm_cooperative(func_as_arg_1, func_as_arg_2, x_1_current, x_2_current, None, None,
                                               grad_1, grad_2, form, beta, 'min', 'min', grad_epsilon, calc_epsilon_x,
                                               calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                               continue_transformation, print_iter_index, **kwargs)

        if len(r_alg_result) == 4:
            grads_1.append(r_alg_result[2][-1])
            grads_2.append(r_alg_result[3][-1])

        x_1_next = r_alg_result[0][-1]
        x_2_next = r_alg_result[1][-1]

        results_1.append(x_1_next.copy())
        results_2.append(x_2_next.copy())

        k += 1

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


# Метод внутренней точки с штрафными функциями вида -rk * sum(i=1, m, 1 / gi(x))
def r_algorithm_interior_point_2(func, x0, g, r_k, args=None, grad=middle_grad_non_matrix, form='B', beta=0.5,
                                 target='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10,
                                 step_epsilon=1e-15, r_epsilon=1e-10, iter_lim=1000000, return_grads=False,
                                 tqdm_fl=False, continue_transformation=True, print_iter_index=False, **kwargs):
    sign = target_input(target)
    x_current, x_next = np.random.rand(x0.size), x0_input(x0)
    k = 0
    results, grads = [x_next.copy()], []
    while np.linalg.norm(x_current - x_next) > calc_epsilon_x and r_k(k) > r_epsilon and k < iter_lim:
        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_current = x_next.copy()

        print([g[i](x_current) for i in range(len(g))])

        if args is None:
            x_next = r_algorithm(lambda x: sign * func(x) -
                                           r_k(k) * np.array([1 / g[i](x) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)
        else:
            x_next = r_algorithm(lambda x: sign * func(x, args) -
                                           r_k(k) * np.array([1 / g[i](x) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)

        if type(x_next) is tuple:
            grads.append(x_next[1].copy())
            x_next = x_next[0].copy()
        x_next = x_next[-1].copy()
        # if print_iter_index:
        #     print(x_next)
        #     if args is None:
        #         print(func(x_next))
        #     else:
        #         print(func(x_next, args))
        results.append(x_next.copy())
        k += 1
        # if print_iter_index:
        #     print('---------------------------')
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_interior_point_2_cooperative(func_1, func_2, x0_1, x0_2, g_1, g_2, r_k_1, r_k_2, args_1=None,
                                             args_2=None, grad_1=middle_grad_non_matrix_pool,
                                             grad_2=middle_grad_non_matrix_pool, form='B', beta=0.5, target_1='min',
                                             target_2='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10,
                                             calc_epsilon_grad=1e-10, step_epsilon=1e-15, r_epsilon=1e-10,
                                             iter_lim=1000000, return_grads=False, tqdm_fl=False,
                                             continue_transformation=True, print_iter_index=False, **kwargs):
    sign_1, sign_2 = target_input(target_1), target_input(target_2)
    x_1_current, x_1_next = np.random.rand(x0_1.size), x0_input(x0_1)
    x_2_current, x_2_next = np.random.rand(x0_2.size), x0_input(x0_2)
    k = 0
    results_1, grads_1 = [x_1_next.copy()], []
    results_2, grads_2 = [x_2_next.copy()], []

    while np.linalg.norm(np.concatenate((x_1_current, x_2_current)) - np.concatenate((x_1_next, x_2_next))) > \
            calc_epsilon_x and r_k_1(k) > r_epsilon and r_k_2(k) > r_epsilon and k < iter_lim:

        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_1_current, x_2_current = x_1_next.copy(), x_2_next.copy()

        if args_1 is None:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y) - r_k_1(k) * \
                                         np.array([1.0 / g_1[i](x, y) for i in range(len(g_1))]).sum()
        else:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y, args_1) - r_k_1(k) * \
                                         np.array([1.0 / g_1[i](x, y) for i in range(len(g_1))]).sum()

        if args_2 is None:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y) - r_k_2(k) * \
                                         np.array([1.0 / g_2[i](x, y) for i in range(len(g_2))]).sum()
        else:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y, args_2) - r_k_2(k) * \
                                         np.array([1.0 / g_2[i](x, y) for i in range(len(g_2))]).sum()

        r_alg_result = r_algorithm_cooperative(func_as_arg_1, func_as_arg_2, x_1_current, x_2_current, None, None,
                                               grad_1, grad_2, form, beta, 'min', 'min', grad_epsilon, calc_epsilon_x,
                                               calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                               continue_transformation, print_iter_index, **kwargs)

        if len(r_alg_result) == 4:
            grads_1.append(r_alg_result[2][-1])
            grads_2.append(r_alg_result[3][-1])

        x_1_next = r_alg_result[0][-1]
        x_2_next = r_alg_result[1][-1]

        results_1.append(x_1_next.copy())
        results_2.append(x_2_next.copy())

        k += 1

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


# Метод внутренней точки с штрафными функциями вида rk * sum(i=1, m, max{-ln(-g[i](x)), 0})
def r_algorithm_interior_point_3(func, x0, g, r_k, args=None, grad=middle_grad_non_matrix, form='B', beta=0.5,
                                 target='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10,
                                 step_epsilon=1e-15, r_epsilon=1e-10, iter_lim=1000000, return_grads=False,
                                 tqdm_fl=False, continue_transformation=True, print_iter_index=False, **kwargs):
    sign = target_input(target)
    x_current, x_next = np.random.rand(x0.size), x0_input(x0)
    k = 0
    results, grads = [x_next.copy()], []
    while np.linalg.norm(x_current - x_next) > calc_epsilon_x and r_k(k) > r_epsilon and k < iter_lim:
        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_current = x_next.copy()

        print([g[i](x_current) for i in range(len(g))])

        if args is None:
            x_next = r_algorithm(lambda x: sign * func(x) + r_k(k) *
                                           np.array([np.maximum(-np.log(-g[i](x)), 0.0) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)
        else:
            x_next = r_algorithm(lambda x: sign * func(x, args) + r_k(k) *
                                           np.array([np.maximum(-np.log(-g[i](x)), 0.0) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)

        if type(x_next) is tuple:
            grads.append(x_next[1].copy())
            x_next = x_next[0].copy()
        x_next = x_next[-1].copy()
        # if print_iter_index:
        #     print(x_next)
        #     if args is None:
        #         print(func(x_next))
        #     else:
        #         print(func(x_next, args))
        results.append(x_next.copy())
        k += 1
        # if print_iter_index:
        #     print('---------------------------')
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_interior_point_3_cooperative(func_1, func_2, x0_1, x0_2, g_1, g_2, r_k_1, r_k_2, args_1=None,
                                             args_2=None, grad_1=middle_grad_non_matrix_pool,
                                             grad_2=middle_grad_non_matrix_pool, form='B', beta=0.5, target_1='min',
                                             target_2='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10,
                                             calc_epsilon_grad=1e-10, step_epsilon=1e-15, r_epsilon=1e-10,
                                             iter_lim=1000000, return_grads=False, tqdm_fl=False,
                                             continue_transformation=True, print_iter_index=False, **kwargs):
    sign_1, sign_2 = target_input(target_1), target_input(target_2)
    x_1_current, x_1_next = np.random.rand(x0_1.size), x0_input(x0_1)
    x_2_current, x_2_next = np.random.rand(x0_2.size), x0_input(x0_2)
    k = 0
    results_1, grads_1 = [x_1_next.copy()], []
    results_2, grads_2 = [x_2_next.copy()], []

    while np.linalg.norm(np.concatenate((x_1_current, x_2_current)) - np.concatenate((x_1_next, x_2_next))) > \
            calc_epsilon_x and r_k_1(k) > r_epsilon and r_k_2(k) > r_epsilon and k < iter_lim:

        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_1_current, x_2_current = x_1_next.copy(), x_2_next.copy()

        if args_1 is None:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y) + \
                                         r_k_1(k) * np.array([np.maximum(-np.log(-g_1[i](x, y)), 0.0)
                                                              for i in range(len(g_1))]).sum()
        else:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y, args_1) + \
                                         r_k_1(k) * np.array([np.maximum(-np.log(-g_1[i](x, y)), 0.0)
                                                              for i in range(len(g_1))]).sum()

        if args_2 is None:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y) + \
                                         r_k_2(k) * np.array([np.maximum(-np.log(-g_2[i](x, y)), 0.0)
                                                              for i in range(len(g_2))]).sum()
        else:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y, args_2) + \
                                         r_k_2(k) * np.array([np.maximum(-np.log(-g_2[i](x, y)), 0.0)
                                                              for i in range(len(g_2))]).sum()

        r_alg_result = r_algorithm_cooperative(func_as_arg_1, func_as_arg_2, x_1_current, x_2_current, None, None,
                                               grad_1, grad_2, form, beta, 'min', 'min', grad_epsilon, calc_epsilon_x,
                                               calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                               continue_transformation, print_iter_index, **kwargs)

        if len(r_alg_result) == 4:
            grads_1.append(r_alg_result[2][-1])
            grads_2.append(r_alg_result[3][-1])

        x_1_next = r_alg_result[0][-1]
        x_2_next = r_alg_result[1][-1]

        results_1.append(x_1_next.copy())
        results_2.append(x_2_next.copy())

        k += 1

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


# Метод внутренней точки с штрафными функциями вида -S * sum(i=1, m, max{g[i](x), 0})
def r_algorithm_interior_point_4(func, x0, g, r_k, args=None, grad=middle_grad_non_matrix, form='B', beta=0.5,
                                 target='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10,
                                 step_epsilon=1e-15, r_epsilon=1e-10, iter_lim=1000000, return_grads=False,
                                 tqdm_fl=False, continue_transformation=True, print_iter_index=False, **kwargs):
    sign = target_input(target)
    x_current, x_next = np.random.rand(x0.size), x0_input(x0)
    k = 0
    results, grads = [x_next.copy()], []
    while np.linalg.norm(x_current - x_next) > calc_epsilon_x and r_k(k) > r_epsilon and k < iter_lim:
        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_current = x_next.copy()

        print([g[i](x_current) for i in range(len(g))])

        if args is None:
            x_next = r_algorithm(lambda x: sign * func(x) + r_k(k) *
                                           np.array([np.maximum(g[i](x), 0.0) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)
        else:
            x_next = r_algorithm(lambda x: sign * func(x, args) + r_k(k) *
                                           np.array([np.maximum(g[i](x), 0.0) for i in range(len(g))]).sum(),
                                 x_current, None, grad, form, beta, 'min', grad_epsilon, calc_epsilon_x,
                                 calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                 continue_transformation,
                                 print_iter_index, **kwargs)

        if type(x_next) is tuple:
            grads.append(x_next[1].copy())
            x_next = x_next[0].copy()
        x_next = x_next[-1].copy()
        # if print_iter_index:
        #     print(x_next)
        #     if args is None:
        #         print(func(x_next))
        #     else:
        #         print(func(x_next, args))
        results.append(x_next.copy())
        k += 1
        # if print_iter_index:
        #     print('---------------------------')
    if return_grads:
        return np.array(results), np.array(grads)
    return np.array(results)


def r_algorithm_interior_point_4_cooperative(func_1, func_2, x0_1, x0_2, g_1, g_2, r_k_1, r_k_2, args_1=None,
                                             args_2=None, grad_1=middle_grad_non_matrix_pool,
                                             grad_2=middle_grad_non_matrix_pool, form='B', beta=0.5, target_1='min',
                                             target_2='min', grad_epsilon=1e-8, calc_epsilon_x=1e-10,
                                             calc_epsilon_grad=1e-10, step_epsilon=1e-15, r_epsilon=1e-10,
                                             iter_lim=1000000, return_grads=False, tqdm_fl=False,
                                             continue_transformation=True, print_iter_index=False, **kwargs):
    sign_1, sign_2 = target_input(target_1), target_input(target_2)
    x_1_current, x_1_next = np.random.rand(x0_1.size), x0_input(x0_1)
    x_2_current, x_2_next = np.random.rand(x0_2.size), x0_input(x0_2)
    k = 0
    results_1, grads_1 = [x_1_next.copy()], []
    results_2, grads_2 = [x_2_next.copy()], []

    while np.linalg.norm(np.concatenate((x_1_current, x_2_current)) - np.concatenate((x_1_next, x_2_next))) > \
            calc_epsilon_x and r_k_1(k) > r_epsilon and r_k_2(k) > r_epsilon and k < iter_lim:

        if print_iter_index:
            print('Индекс итерации метода внутренней точки: %d' % k)

        x_1_current, x_2_current = x_1_next.copy(), x_2_next.copy()

        if args_1 is None:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y) + \
                                         r_k_1(k) * np.array([np.maximum(g_1[i](x, y), 0.0)
                                                              for i in range(len(g_1))]).sum()
        else:
            func_as_arg_1 = lambda x, y: sign_1 * func_1(x, y, args_1) + \
                                         r_k_1(k) * np.array([np.maximum(g_1[i](x, y), 0.0)
                                                              for i in range(len(g_1))]).sum()

        if args_2 is None:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y) + \
                                         r_k_2(k) * np.array([np.maximum(g_2[i](x, y), 0.0)
                                                              for i in range(len(g_2))]).sum()
        else:
            func_as_arg_2 = lambda x, y: sign_2 * func_2(x, y, args_2) + \
                                         r_k_2(k) * np.array([np.maximum(g_2[i](x, y), 0.0)
                                                              for i in range(len(g_2))]).sum()

        r_alg_result = r_algorithm_cooperative(func_as_arg_1, func_as_arg_2, x_1_current, x_2_current, None, None,
                                               grad_1, grad_2, form, beta, 'min', 'min', grad_epsilon, calc_epsilon_x,
                                               calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl,
                                               continue_transformation, print_iter_index, **kwargs)

        if len(r_alg_result) == 4:
            grads_1.append(r_alg_result[2][-1])
            grads_2.append(r_alg_result[3][-1])

        x_1_next = r_alg_result[0][-1]
        x_2_next = r_alg_result[1][-1]

        results_1.append(x_1_next.copy())
        results_2.append(x_2_next.copy())

        k += 1

    if return_grads:
        return np.array(results_1), np.array(results_2), np.array(grads_1), np.array(grads_2)

    return np.array(results_1), np.array(results_2)


# Функция вычисляющая двойной интеграл методом трапеций
def trapezoid_double(integrand, x_a, x_b, y_a, y_b, grid_dot_num_x=10, grid_dot_num_y=10):
    x_vals, y_vals = np.linspace(x_a, x_b, grid_dot_num_x + 1), np.linspace(y_a, y_b, grid_dot_num_y + 1)
    xx, yy = np.meshgrid(x_vals, y_vals)
    integrand_vals = integrand(xx, yy)
    return (x_b - x_a) * (y_b - y_a) / 4 / grid_dot_num_x / grid_dot_num_y * \
           (integrand_vals[:grid_dot_num_y, :grid_dot_num_x].sum() + integrand_vals[1:, :grid_dot_num_x].sum() +
            integrand_vals[:grid_dot_num_y, 1:].sum() + integrand_vals[1:, 1:].sum())


# Функция вычисляющая двойной интеграл методом Гаусса
def integral_double(integrand, x_a, x_b, y_a, y_b, grid_dot_num_x=10, grid_dot_num_y=10):

    legc_x, legc_y = np.zeros(grid_dot_num_x + 2), np.zeros(grid_dot_num_y + 2)
    legc_x[-1], legc_y[-1] = 1.0, 1.0

    x_vals, y_vals = leg.legroots(legc_x), leg.legroots(legc_y)

    legc_x[-1], legc_y[-1] = 0.0, 0.0
    legc_x[-2], legc_y[-2] = 1.0, 1.0
    weights_x, weights_y = \
        2 / ((grid_dot_num_x + 1) * leg.legval(x_vals, legc_x)) ** 2 * (1 - x_vals ** 2), \
        2 / ((grid_dot_num_y + 1) * leg.legval(y_vals, legc_y)) ** 2 * (1 - y_vals ** 2)

    xx, yy = np.meshgrid(x_vals, y_vals)
    weights_xx, weights_yy = np.meshgrid(weights_x, weights_y)

    return ((x_b - x_a) * (y_b - y_a) / 4 * weights_xx * weights_yy *
            integrand((x_b + x_a) / 2 + (x_b - x_a) / 2 * xx, (y_b + y_a) / 2 + (y_b - y_a) / 2 * yy)).sum()


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


# Целевой функционал задачи А6
# Принимает те же аргументы, что и двойственный функционал данной задачи (linear_partition_problem_target_dual)
def linear_partition_problem_target(psi, tau, args):

    tau = tau_transformation_from_vector_to_matrix(tau)

    partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right, \
    y_left, y_right, grid_dot_num_x, grid_dot_num_y = args

    if partition_number != tau.shape[1] or partition_number != b_vector.size or \
            partition_number != a_matrix.shape[0] or product_number != len(cost_function_vector) or \
            product_number != len(density_vector) or product_number != a_matrix.shape[1]:
        raise ValueError('Please, check input data!')

    return np.array([
        integral_double(lambda x, y: np.array(
            (cost_function_vector[j](x, y, tau) +
             a_matrix[:, j].reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1]))) *
            np.where(
                cost_function_vector[j](x, y, tau) +
                a_matrix[:, j].reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1])) +
                psi.reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1])) ==
                np.array(
                    cost_function_vector[j](x, y, tau) +
                    a_matrix[:, j].reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1]))
                    + psi.reshape(partition_number, 1, 1) * np.ones((partition_number, x.shape[0], x.shape[1]))
                ).min(axis=0), 1.0, 0.0
            )
        ).sum(axis=0) * density_vector[j](x, y), x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y)
        for j in range(product_number)]).sum()


# Двойственный функционал задачи А6 (Линейной многопродуктовой задачи ОРМ с размещением центров)
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
def linear_partition_problem_target_dual(psi, tau, args):
    tau = tau_transformation_from_vector_to_matrix(tau)

    partition_number, product_number, cost_function_vector, density_vector, a_matrix, b_vector, x_left, x_right, \
    y_left, y_right, grid_dot_num_x, grid_dot_num_y = args

    if partition_number != tau.shape[1] or partition_number != psi.size or partition_number != b_vector.size or \
            partition_number != a_matrix.shape[0] or product_number != len(cost_function_vector) or \
            product_number != len(density_vector) or product_number != a_matrix.shape[1]:
        raise ValueError('Please, check input data!')

    return np.array([
        integral_double(
            lambda x, y: np.array(cost_function_vector[j](x, y, tau) +
                                  a_matrix[:, j].reshape(partition_number, 1, 1) *
                                  np.ones((partition_number, x.shape[0], x.shape[1])) +
                                  psi.reshape(partition_number, 1, 1) *
                                  np.ones((partition_number, x.shape[0], x.shape[1]))).min(axis=0) *
            density_vector[j](x, y), x_left, x_right, y_left, y_right, grid_dot_num_x, grid_dot_num_y)
        for j in range(product_number)]).sum() - np.dot(psi, b_vector)


# Двойственный функционал задачи А6 (Линейной многопродуктовой задачи ОРМ с размещением центров) с барьерными функциями
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
# additional_args -- параметр, через который будут передаваться дополнительные параметры целевого функционала,
# не передающиеся в функционал без барьерных функций
# psi_limitations -- ограничения по psi
# tau_limitations -- ограничения по tau
# psi_penalty -- штраф по psi
# tau_penalty -- штраф по tau
def linear_partition_problem_target_dual_interior_point(psi, tau, args, additional_args):

    psi_limitations, tau_limitations, psi_penalty, tau_penalty = additional_args

    return linear_partition_problem_target_dual(psi, tau, args) - psi_penalty *\
           np.array([np.maximum(psi_limitations[i](psi), 0.0) for i in range(len(psi_limitations))]).sum() +\
           tau_penalty * np.array([np.maximum(tau_limitations[i](tau), 0.0) for i in range(len(tau_limitations))]).sum()
