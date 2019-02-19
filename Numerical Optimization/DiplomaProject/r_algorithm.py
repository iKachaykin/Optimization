import numpy as np
import numpy.linalg as linalg
from math import isnan
from tqdm import tqdm


# Вспомогательная функция, реализующая алгоритм метода золотого сечения
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# func - целевая функция
# a - левый край промежутка, на котором выполняется поиск
# b - правый край промежутка, на котором выполняется поиск
# target_dual - строковая переменная, отвечающая за тип задачи оптимизации (минимизация или максимизация);
# допустимые значения:
# 1) min или minimum - для минимума
# 2) max или maximum - для максимума
# epsilon - точность вычислений
# iter_lim - предельное количество итераций
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение: приближенное значение точки, в которой функция достигает наименьшего (наибольшего) значения
def gsection(func, a, b, target="min", epsilon=1e-10, iter_lim=1000000):
    if a >= b:
        a, b = b, a
    sign = target_input(target)
    multiplier1, multiplier2 = (3.0 - np.sqrt(5)) / 2.0, (np.sqrt(5) - 1.0) / 2.0
    dot1, dot2 = a + multiplier1 * (b - a), a + multiplier2 * (b - a)
    counter = 0
    while b - a > epsilon and counter < iter_lim:
        if sign * func(dot1) > sign * func(dot2):
            a, dot1, dot2 = dot1, dot2, dot1 + multiplier2 * (b - dot1)
        else:
            b, dot1, dot2 = dot2, a + multiplier1 * (dot2 - a), dot1
        counter += 1
    return (a + b) / 2.0

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
    gradient, unit_m = np.zeros_like(x0), np.eye(x0.size, x0.size)
    for i in range(x0.size):
        gradient[i] = (func(x0 + epsilon * unit_m[i]) - func(x0 - epsilon * unit_m[i])) / 2 / epsilon
    return gradient


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
        grad = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('default_step'), \
        kwargs.get('step_red_mult'), kwargs.get('step_incr_mult'), kwargs.get('lim_num'), \
        kwargs.get('reduction_epsilon'), kwargs.get('step_epsilon'), kwargs.get('grad')
    step = default_step
    while reduction_epsilon >= func(x_current) - func(x_current - step * direction) and np.abs(step) > step_epsilon:
        step *= step_red_mult
    break_flag = 0
    tmp_step, step = step, 0.0
    while step < 10.0:
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
        # print(step)
    if break_flag == 2:
        tmp_step /= step_incr_mult
    return step, tmp_step


def step_adaptive_alternative(kwargs):
    func, x_current, direction, default_step, step_red_mult, step_incr_mult, lim_num, step_lim = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('default_step'), \
        kwargs.get('step_red_mult'), kwargs.get('step_incr_mult'), kwargs.get('lim_num'), kwargs.get('step_lim')
    step = default_step
    i = 0
    while func(x_current - i * step * direction) > func(x_current - (i + 1) * step * direction) and i < step_lim:
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
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение:
# 1) если return_grads = True, то возвращаемое значение - tuple, первый элемент которой - list всех точек приближения,
# второй - list всех субградиентов в соответствующих точках
# 2) если return_grads = False, то возвращаемое значение - list всех точек приближения
def r_algorithm_B_form(func, x0, grad, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                       calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl, continue_transformation):
    x_current, x_next, matrix_B, grad_current, grad_next = \
        x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
        grad(x0, func, epsilon=grad_epsilon), middle_grad_non_matrix(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']
    step_method_kwargs['func'] = func
    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad'] = grad
    results = [x_next.copy()]
    grads = [grad_next.copy()]
    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)
    for k in iterations:
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
    return matrix_H + (beta * beta - 1) \
                      * np.dot(np.dot(matrix_H, r_vector).reshape(r_vector.size, 1),
                               np.dot(matrix_H, r_vector).reshape(1, r_vector.size)) \
                      / np.dot(np.dot(r_vector, matrix_H), r_vector)


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
# ----------------------------------------------------------------------------------------------------------------------
# Возвращаемое значение:
# 1) если return_grads = True, то возвращаемое значение - tuple, первый элемент которой - list всех точек приближения,
# второй - list всех субградиентов в соответствующих точках
# 2) если return_grads = False, то возвращаемое значение - list всех точек приближения
def r_algorithm_H_form(func, x0, grad, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon_x,
                       calc_epsilon_grad, step_epsilon, iter_lim, return_grads, tqdm_fl, continue_transformation):
    x_current, x_next, matrix_H, grad_current, grad_next = \
        x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
        grad(x0, func, epsilon=grad_epsilon), grad(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive, 'adaptive_alternative': step_adaptive_alternative}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive', 'adaptive_alternative']
    step_method_kwargs['func'] = func
    step_method_kwargs['step_lim'] = iter_lim
    step_method_kwargs['grad'] = grad
    results = [x_next.copy()]
    grads = [grad_next.copy()]
    if tqdm_fl:
        iterations = tqdm(range(iter_lim))
    else:
        iterations = range(iter_lim)
    for k in iterations:
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
def r_algorithm(func, x0, args=None, grad=middle_grad_non_matrix, form='B', beta=0.5, target='min', grad_epsilon=1e-8,
                calc_epsilon_x=1e-10, calc_epsilon_grad=1e-10, step_epsilon=1e-15, iter_lim=1000000, return_grads=False,
                tqdm_fl=False, continue_transformation=True, **kwargs):
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
                                  continue_transformation=continue_transformation)
    else:
        return r_algorithm_B_form(func_as_arg, x0, grad, beta, step_method, step_method_kwargs,
                                  grad_epsilon=grad_epsilon, calc_epsilon_x=calc_epsilon_x,
                                  calc_epsilon_grad=calc_epsilon_grad, step_epsilon=step_epsilon, iter_lim=iter_lim,
                                  return_grads=return_grads, tqdm_fl=tqdm_fl,
                                  continue_transformation=continue_transformation)


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
