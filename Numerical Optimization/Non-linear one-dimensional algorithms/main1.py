import NonLinearOptimization as nlopt
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 1 / (1 + x*x)

a, b, dot_num, additional_length_multiplier_x, additional_length_multiplier_y, plot_style, borders_style, min_style, \
max_style, figsize, methods_number \
    = -np.pi, np.pi, 1000, 0.1, 0.0, 'k-', 'k--', 'ro', 'bo', (15, 7.5), 2
optimization_methods = (nlopt.dichotomy, nlopt.gsection)
optimization_methods_names = ("Метод дихотомии", "Метод золотого сечения")
x = np.linspace(a - additional_length_multiplier_x * (b - a), b + additional_length_multiplier_x * (b - a))
y = f(x)
for i in range(methods_number):
    argmin = optimization_methods[i](f, a, b, target="min")
    argmax = optimization_methods[i](f, a, b, target="max")
    tmp = plt.figure(num=i+1, figsize=figsize)
    plt.suptitle(optimization_methods_names[i])
    plt.plot(x, y, plot_style, argmin, f(argmin), min_style, argmax, f(argmax), max_style,
         [a, a], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                  np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style,
         [b, b], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                  np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style)
    plt.grid(True)
plt.show()
plt.close()
