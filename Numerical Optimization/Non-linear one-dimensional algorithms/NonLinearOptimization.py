import numpy as np
from scipy.misc import derivative


def dichotomy(func, a, b, target="min", epsilon=1e-10, iter_lim=1000000):
    if np.abs(epsilon) < 2e-15:
        raise "epsilon is to small to perform calculations"
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise "invalid value of \"target\""
    delta = epsilon / 2
    counter = 0
    while b - a > epsilon and counter < iter_lim:
        if sign * func((a + b - delta) / 2.0) <= sign * func((a + b + delta) / 2.0):
            b = (a + b + delta) / 2.0
        else:
            a = (a + b - delta) / 2.0
        counter += 1
    return (a + b) / 2.0


def gsection(func, a, b, target="min", epsilon=1e-10, iter_lim=1000000):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise "invalid value of \"target\""
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


def fibonacci(func, a, b, target="min", epsilon=1e-10, iter_lim=1000000):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise "invalid value of \"target\""
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
    return (a + b) / 2.0


def tangent(func, a, b, target="min", epsilon=1e-10, iter_lim=1000000, dx=1e-3):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise "invalid value of \"target\""
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
        raise "invalid value of \"target\""
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
    counter = 0
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
