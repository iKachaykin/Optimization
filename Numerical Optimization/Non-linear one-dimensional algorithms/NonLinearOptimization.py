import numpy as np
from scipy.misc import derivative


def dichotomy(func, a, b, target="min", epsilon=1e-10):
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
    delt = epsilon / 2
    while b - a > epsilon:
        if sign * func((a + b - delt) / 2.0) <= sign * func((a + b + delt) / 2.0):
            b = (a + b + delt) / 2.0
        else:
            a = (a + b - delt) / 2.0
    return (a + b) / 2.0


def gsection(func, a, b, target="min", epsilon=1e-10):
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
    while b - a > epsilon:
        if sign * func(dot1) > sign * func(dot2):
            a, dot1, dot2 = dot1, dot2, dot1 + multiplier2 * (b - dot1)
        else:
            b, dot1, dot2 = dot2, a + multiplier1 * (dot2 - a), dot1
    return (a + b) / 2.0


def fibonacci(func, a, b, target="min", epsilon=1e-10):
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
    while (b - a) / epsilon > fib_sequence[fib_number + 2]:
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


def tangent(func, a, b, target="min", epsilon=1e-10, dx=1e-3):
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise "invalid value of \"target\""
    while (b - a) / 2.0 > epsilon:
        x = (b * sign * derivative(func, b, dx=dx) - a * sign * derivative(func, a, dx=dx) +
             sign * func(a) - sign * func(b)) / (sign * derivative(func, b, dx=dx) - sign * derivative(func, a, dx=dx))
        if np.abs(derivative(func, x, dx=dx)) < epsilon:
            return x
        elif sign * derivative(func, x, dx=dx) > epsilon:
            b = x
        else:
            a = x
    return (a + b) / 2.0