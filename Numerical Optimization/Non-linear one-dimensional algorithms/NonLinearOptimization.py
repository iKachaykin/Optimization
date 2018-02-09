import numpy as np


def dichotomy(func, a, b, target="min", epsilon=2e-10, delt=1e-10):
    if epsilon < delt:
        raise "epsilon had to be greater than delt"
    if a >= b:
        a, b = b, a
    if target.lower() == "min" or target.lower() == "minimum":
        sign = 1.0
    elif target.lower() == "max" or target.lower() == "maximum":
        sign = -1.0
    else:
        raise "invalid value of \"target\""
    while b - a > epsilon:
        if sign * func((a + b - delt) / 2.0) <= sign * func((a + b + delt) / 2.0):
            b = (a + b + delt) / 2.0
        else:
            a = (a + b - delt) / 2.0
    return (b + a) / 2.0


def gsection(func, a, b, target="min", epsilon=2e-10):
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