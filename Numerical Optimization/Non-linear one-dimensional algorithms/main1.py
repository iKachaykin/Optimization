import NonLinearOptimization as nlopt
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x * x + 2 * x

a, b, dot_num, additional_length_multiplier_x, additional_length_multiplier_y, plot_style, borders_style, min_style, \
max_style, figsize, methods_number, argmin_exact, argmax_exact, data \
    = -3.0, 5.0, 1000, 0.25, 0.0, 'k-', 'k--', 'ro', 'bo', (15, 7.5), 4, -1.0, 5.0, []
optimization_methods = (nlopt.dichotomy, nlopt.gsection, nlopt.fibonacci, nlopt.tangent)
optimization_methods_names = ("Метод дихотомии", "Метод золотого сечения", "Метод Фибоначчи", "Метод касательных")
table_head = ("Название метода", "Численное решение (min)", "Численное решение (max)", "Точное решение (min)",
              "Точное решение (max)", "Абсолютная погрешность (min)", "Абсолютная погрешность (max)")
x = np.linspace(a - additional_length_multiplier_x * (b - a), b + additional_length_multiplier_x * (b - a), dot_num)
y = f(x)
for i in range(methods_number):
    argmin_numerical = optimization_methods[i](f, a, b, target="min")
    argmax_numerical = optimization_methods[i](f, a, b, target="max", epsilon=1e-6)
    plt.figure(num=i+1, figsize=figsize)
    plt.suptitle(optimization_methods_names[i])
    plt.plot(x, y, plot_style, argmin_numerical, f(argmin_numerical), min_style, argmax_numerical, f(argmax_numerical), max_style,
             [a, a], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                  np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style,
             [b, b], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                  np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style)
    plt.grid(True)
    data.append([optimization_methods_names[i], argmin_numerical, argmax_numerical, argmin_exact, argmax_exact,
                 np.abs(argmin_numerical - argmin_exact), np.abs(argmax_numerical - argmax_exact)])
plt.figure(num=methods_number+1, figsize=figsize)
plt.axis('off')
plt.table(cellText=data, loc='center', colLabels=table_head, cellLoc='center')
plt.show()
plt.close()
