import NonLinearOptimization as nlopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


def f(x):
    return x / (x * x + 1)

a, b, dot_num, additional_length_multiplier_x, additional_length_multiplier_y, plot_style, borders_style, solution_style, \
figsize, methods_number, solution_exact, data, target \
    = -1.5, 0.0, 1000, 0.25, 0.0, 'k-', 'k--', 'ro', (15.0, 7.5), 4, -1.0, [], "min"
a_print, b_print = a, b
optimization_methods = (nlopt.dichotomy, nlopt.gsection, nlopt.fibonacci, nlopt.tangent)
optimization_methods_names = ("Метод дихотомии", "Метод золотого сечения", "Метод Фибоначчи", "Метод касательных")
table_head = ("Название\nметода", "Численное\nрешение", "Значение\nцелевой\nфункции",
              "Абсолютная\nпогрешность\n(аргумент)", "Абсолютная\nпогрешность\n(функция)")
x = np.linspace(a - additional_length_multiplier_x * (b - a), b + additional_length_multiplier_x * (b - a), dot_num)
y = f(x)
ncols, nrows = len(table_head), methods_number + 1
for i in range(methods_number):
    solution_numerical = optimization_methods[i](f, a, b, target=target, epsilon=1e-6)
    plt.figure(num=i+1, figsize=figsize)
    plt.title(optimization_methods_names[i])
    plt.plot(x, y, plot_style, solution_numerical, f(solution_numerical), solution_style,
             [a, a], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                  np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style,
             [b, b], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                  np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style)
    plt.grid(True)
    data.append([optimization_methods_names[i], solution_numerical, f(solution_numerical),
                 np.abs(solution_numerical - solution_exact), np.abs(f(solution_numerical) - f(solution_exact))])
fig = plt.figure(num=methods_number+1, figsize=figsize)
ax = plt.subplot()
ax.axis('off')
tab = Table(ax, bbox=[0, 0, 1, 1])
tab.auto_set_column_width(False)
tab.auto_set_font_size(False)
for j in range(ncols):
    tab.add_cell(0, j, figsize[0] / ncols, 0.1, text=table_head[j], loc="center")
for i in range(1, nrows):
    for j in range(ncols):
        tab.add_cell(i, j, figsize[0] / ncols, 0.1, text=str(data[i-1][j]), loc="center")
tab.set_fontsize(9.0)
ax.add_table(tab)
plt.title(r"$Задача: f(x) = \frac{x}{x^2 + 1} \rightarrow %s, x\in\left[%f; %f\right]$"
          r"%sТочное решение: $x_{%s} = %.10f; f(x_{%s}) = %.10f$"
          % (target, a, b, "\n\n", target, solution_exact, target, f(solution_exact)))
plt.show()
plt.close()
