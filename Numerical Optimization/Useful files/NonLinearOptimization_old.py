import numpy as np
import numpy.linalg as linalg
from scipy.misc import derivative
from math import isnan


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
        raise ValueError("invalid value of \"target\"")
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
        raise ValueError("invalid value of \"target\"")
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
        raise ValueError("invalid value of \"target\"")
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
        raise ValueError("invalid value of \"target\"")
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
        raise ValueError("invalid value of \"target\"")
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


def left_side_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1)) -
            func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) - epsilon * np.eye(x0.size))) / \
           epsilon


def right_side_grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) + epsilon * np.eye(x0.size)) -
            func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1))) / epsilon


def grad(x0, func, epsilon=1e-6):
    return (func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) + epsilon * np.eye(x0.size)) -
            func(np.ones((x0.size, x0.size)) * x0.reshape(x0.size, 1) - epsilon * np.eye(x0.size))) / \
           2 / epsilon


def gauss(func, x0, alpha_a, alpha_b, target="min", epsilon=1e-10, iter_lim=1000000):
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise ValueError("invalid value of \"target\"")
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
        raise ValueError("invalid value of \"target\"")
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
        raise ValueError("invalid value of \"target\"")
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
        while func(x_current - step * grad(x_current, func, epsilon=grad_epsilon)) >= func(x_current):
            step *= step_red_mult
        # print("%.6f" % step)
        x_next = x_current - step * grad(x_current, func, epsilon=grad_epsilon)
        results.append(x_next.copy())
        if np.abs(func(x_next) - func(x_current)) < epsilon:
            break
        counter += 1
    return np.array(results)


def step_argmin(kwargs):
    func, x_current, direction, step_min, step_max, argmin_finder = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('step_min'), \
        kwargs.get('step_max'), kwargs.get('argmin_finder')
    return argmin_finder(lambda step: func(x_current - step * direction), step_min, step_max)


def step_func(kwargs):
    step_defining_func, step_index = kwargs.get('step_defining_func'), kwargs.get('step_index')
    return step_defining_func(step_index)


def step_reduction(kwargs):
    func, x_current, direction, default_step, step_red_mult, reduction_epsilon, step_epsilon = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('default_step'), \
        kwargs.get('step_red_mult'), kwargs.get('reduction_epsilon'), kwargs.get('step_epsilon')
    step = default_step
    while reduction_epsilon >= func(x_current) - func(x_current - step * direction) and np.abs(step) > step_epsilon:
        step *= step_red_mult
    return step


def step_adaptive(kwargs):
    func, x_current, direction, default_step, step_red_mult, step_incr_mult, lim_num, reduction_epsilon, step_epsilon = \
        kwargs.get('func'), kwargs.get('x_current'), kwargs.get('direction'), kwargs.get('default_step'), \
        kwargs.get('step_red_mult'), kwargs.get('step_incr_mult'), kwargs.get('lim_num'), \
        kwargs.get('reduction_epsilon'), kwargs.get('step_epsilon')
    step = default_step
    while reduction_epsilon >= func(x_current) - func(x_current - step * direction) and np.abs(step) > step_epsilon:
        step *= step_red_mult
    break_flag = 0
    tmp_step, step = step, 0.0
    while True:
        for i in range(1, lim_num + 1):
            tmp1, tmp2 = func(x_current - (step + i * tmp_step) * direction), func(x_current - (step + (i - 1) * tmp_step) * direction)
            if reduction_epsilon >= func(x_current - (step + (i - 1) * tmp_step) * direction) - func(x_current - (step + i * tmp_step) * direction)\
                    or isnan(func(x_current - (step + (i - 1) * tmp_step) * direction))\
                    or isnan(func(x_current - (step + i * tmp_step) * direction)):
                step += (i - 1) * tmp_step
                break_flag = 1 if i != 1 else 2
                break
        if break_flag == 1 or break_flag == 2:
            break
        step += lim_num * tmp_step
        tmp_step *= step_incr_mult
    if break_flag == 2:
        tmp_step /= step_incr_mult
    return step, tmp_step


def matrix_B_transformation(matrix_B, grad_current, grad_next, beta, k=0):
    r_vector = np.dot(matrix_B.T, grad_next - grad_current)
    r_vector = r_vector / linalg.norm(r_vector)
    tmp = np.eye(matrix_B.shape[0], matrix_B.shape[1]) + \
          (beta - 1) * np.dot(r_vector.reshape(r_vector.size, 1), r_vector.reshape(1, r_vector.size))
    return np.dot(matrix_B, np.eye(matrix_B.shape[0], matrix_B.shape[1]) +
                  (beta - 1) * np.dot(r_vector.reshape(r_vector.size, 1), r_vector.reshape(1, r_vector.size)))


def r_algorithm_B_form(func, x0, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon, step_epsilon,
                       iter_lim):
    x_current, x_next, matrix_B, grad_current, grad_next = \
        x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
        left_side_grad(x0, func, epsilon=grad_epsilon), right_side_grad(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive']
    step_method_kwargs['func'] = func
    results = [x_next.copy()]
    for k in range(iter_lim):
        xi_current = np.dot(matrix_B.T, grad_next)
        xi_current = xi_current / linalg.norm(xi_current)
        step_method_kwargs['x_current'] = x_next
        step_method_kwargs['direction'] = np.dot(matrix_B, xi_current)
        step_method_kwargs['step_index'] = k
        step_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)
        if isinstance(step_current, tuple):
            step_current, step_method_kwargs['default_step'] = step_current
        if np.abs(step_current) < step_epsilon and step_method in continuing_step_methods:
            matrix_B = matrix_B_transformation(matrix_B, grad_current, grad_next, beta)
            print('Step to small! Step = %f' % step_current)
            continue
        print('%.3f' % step_current)
        # print('(\matrix(%.3f@%.3f))' % (xi_current[0], xi_current[1]))
        x_current, grad_current = x_next.copy(), grad_next.copy()
        x_next = x_current - step_current * np.dot(matrix_B, xi_current)
        results.append(x_next.copy())
        grad_next = grad(x_next, func, epsilon=grad_epsilon)
        if linalg.norm(grad_next) < calc_epsilon or linalg.norm(x_next - x_current) < calc_epsilon:
            break
        matrix_B = matrix_B_transformation(matrix_B, grad_current, grad_next, beta, k)
    return np.array(results)


def matrix_H_transformation(matrix_H, grad_current, grad_next, beta):
    r_vector = grad_next - grad_current
    return matrix_H + (beta * beta - 1) \
                      * np.dot(np.dot(matrix_H, r_vector).reshape(r_vector.size, 1),
                               np.dot(matrix_H, r_vector).reshape(1, r_vector.size)) \
                      / np.dot(np.dot(r_vector, matrix_H), r_vector)


def r_algorithm_H_form(func, x0, beta, step_method, step_method_kwargs, grad_epsilon, calc_epsilon, step_epsilon,
                       iter_lim):
    x_current, x_next, matrix_H, grad_current, grad_next = \
        x0.copy(), x0.copy(), np.eye(x0.size, x0.size), \
        left_side_grad(x0, func, epsilon=grad_epsilon), right_side_grad(x0, func, epsilon=grad_epsilon)
    step_defining_algorithms = {'argmin': step_argmin, 'func': step_func, 'reduction': step_reduction,
                                'adaptive': step_adaptive}
    continuing_step_methods = ['argmin', 'reduction', 'adaptive']
    step_method_kwargs['func'] = func
    results = [x_next.copy()]
    for k in range(iter_lim):
        step_method_kwargs['x_current'] = x_next
        step_method_kwargs['direction'] = np.dot(matrix_H, grad_next) / \
                                          np.sqrt(np.dot(np.dot(matrix_H, grad_next), grad_next))
        step_method_kwargs['step_index'] = k
        step_current = (step_defining_algorithms.get(step_method))(step_method_kwargs)
        if isinstance(step_current, tuple):
            step_current, step_method_kwargs['default_step'] = step_current
        if np.abs(step_current) < step_epsilon and step_method in continuing_step_methods:
            matrix_H = matrix_H_transformation(matrix_H, grad_current, grad_next, beta)
            continue
        x_current, grad_current = x_next.copy(), grad_next.copy()
        x_next = x_current - step_current * np.dot(matrix_H, grad_current) / \
                             np.sqrt(np.dot(np.dot(matrix_H, grad_current), grad_current))
        results.append(x_next.copy())
        grad_next = grad(x_next, func, epsilon=grad_epsilon)
        if linalg.norm(x_next - x_current) < calc_epsilon or linalg.norm(grad_next) < calc_epsilon:
            break
        matrix_H = matrix_H_transformation(matrix_H, grad_current, grad_next, beta)
    return np.array(results)


def target_input(target):
    if target.lower() == "min" or target.lower() == "minimum":
        return 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        return -1.0
    else:
        raise ValueError("invalid value of \"target\"")


def x0_input(x0):
    if not (isinstance(x0, np.ndarray) or isinstance(x0, list) or isinstance(x0, float)):
        raise TypeError("x0 had to be an array")
    return np.array(x0)


def r_algorithm(func, x0, form='B', beta=0.5, target='min', grad_epsilon=1e-8, calc_epsilon=1e-10,
                step_epsilon=1e-15, iter_lim=1000000, **kwargs):
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
    if 'H' in form:
        return r_algorithm_H_form(lambda x: sign * func(x), x0, beta, step_method, step_method_kwargs,
                                  grad_epsilon=grad_epsilon, calc_epsilon=calc_epsilon, step_epsilon=step_epsilon,
                                  iter_lim=iter_lim)
    else:
        return r_algorithm_B_form(lambda x: sign * func(x), x0, beta, step_method, step_method_kwargs,
                                  grad_epsilon=grad_epsilon, calc_epsilon=calc_epsilon, step_epsilon=step_epsilon,
                                  iter_lim=iter_lim)
