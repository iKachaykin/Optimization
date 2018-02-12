import NonLinearOptimization as nlopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


def f(x):
    return np.abs(x) + 2 * x * x


def main():
    a, b, dot_num, additional_length_multiplier_x, additional_length_multiplier_y, plot_style, borders_style, \
    solution_style, figsize, methods_number, solution_exact, data, target, epsilon, iter_lim \
        = -0.5, 2.0, 1000, 0.0, 0.0, 'k-', 'k--', 'ro', (15.0, 7.5), 5, 0.0, [], "min", 1e-6, 1000
    optimization_methods = (nlopt.dichotomy, nlopt.gsection, nlopt.fibonacci, nlopt.tangent, nlopt.parabolic)
    optimization_methods_names = ("Метод дихотомии", "Метод золотого сечения", "Метод Фибоначчи", "Метод касательных",
                                  "Метод парабол")
    table_head = ("Название\nметода", "Численное\nрешение", "Значение\nцелевой\nфункции",
                  "Абсолютная\nпогрешность\n(аргумент)", "Абсолютная\nпогрешность\n(функция)")
    fmt_a, fmt_b, fmt_solution_exact, fmt_target_value = r"%.", r"%.", r"%.", r"%."
    fmt_a += r"%sf" % str(main_digits_num(a, int(-np.log10(epsilon))))
    fmt_b += r"%sf" % str(main_digits_num(b, int(-np.log10(epsilon))))
    fmt_solution_exact += r"%sf" % str(main_digits_num(solution_exact, int(-np.log10(epsilon))))
    fmt_target_value += r"%sf" % str(main_digits_num(f(solution_exact), int(-np.log10(epsilon))))
    x = np.linspace(a - additional_length_multiplier_x * (b - a), b + additional_length_multiplier_x * (b - a), dot_num)
    y = f(x)
    ncols, nrows = len(table_head), methods_number + 1
    for i in range(methods_number):
        solution_numerical = optimization_methods[i](f, a, b, target=target, epsilon=epsilon, iter_lim=iter_lim)
        plt.figure(num=i + 1, figsize=figsize)
        plt.title(optimization_methods_names[i])
        plt.plot(x, y, plot_style, solution_numerical, f(solution_numerical), solution_style,
                 [a, a], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                          np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style,
                 [b, b], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                          np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style)
        plt.grid(True)
        data.append([optimization_methods_names[i], solution_numerical, f(solution_numerical),
                     np.abs(solution_numerical - solution_exact), np.abs(f(solution_numerical) - f(solution_exact))])
    fig = plt.figure(num=methods_number + 1, figsize=figsize)
    ax = plt.subplot()
    ax.axis('off')
    tab = Table(ax, bbox=[0, 0, 1, 1])
    tab.auto_set_column_width(False)
    tab.auto_set_font_size(False)
    for j in range(ncols):
        tab.add_cell(0, j, figsize[0] / ncols, 0.1, text=table_head[j], loc="center")
    for i in range(1, nrows):
        for j in range(ncols):
            tab.add_cell(i, j, figsize[0] / ncols, 0.1, text=str(data[i - 1][j]), loc="center")
    tab.set_fontsize(9.0)
    ax.add_table(tab)
    plt.title(r"$Задача: f(x) = |x| + 2x^2 \rightarrow %s, x\in[$" % target + fmt_a % a + r"; " + fmt_b % b
              + r"]%sТочное решение:" % "\n\n" + r"$x_{%s}$ = " % target + fmt_solution_exact % solution_exact +
              r"; $f(x_{%s})$ = " % target + fmt_target_value % f(solution_exact))
    plt.show()
    plt.close()


def main_digits_num(number, digits_to_check=10):
    if not isinstance(number, float):
        raise "number had to be float"
    number = np.abs(number)
    digits, tmp = \
        np.zeros(digits_to_check, dtype=np.int64), np.int64((number - int(number)) * np.power(10, digits_to_check))
    for i in range(digits_to_check):
        digits[i] = tmp % 10
        tmp //= 10
    count = digits_to_check
    for i in range(digits_to_check):
        if digits[i] != 0:
            return count
        count -= 1
    return count


if __name__ == '__main__':
    main()
