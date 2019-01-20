import NonLinearOptimization_old as nlopt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table


def f(x):
    return x / (x * x + 1)


def main():
    a, b, a_lst, b_lst, dot_num, additional_length_multiplier_x, additional_length_multiplier_y, plot_style, \
    borders_style, solution_style, a_style, b_style, figsize, methods_number, solution_exact, data, target, epsilon, iter_lim \
        = -1.5, 0.0, [], [], 1000, 0.0, 0.0, 'k-', 'k--', 'ko', '<k', 'k>', (15.0, 7.5), 3, -1.0, [], "min", 1e-3, 1000
    optimization_methods = (nlopt.dichotomy, nlopt.gsection, nlopt.fibonacci, nlopt.tangent, nlopt.parabolic)
    optimization_methods_names = (u"Метод дихотомії", u"Метод золотого перерізу", u"Метод Фібоначчи", u"Метод дотичних",
                                  u"Метод парабол")
    table_head = (u"Назва\nметоду", u"Чисельний\nрозв'язок", u"Значення\nцільової\nфункції",
                  u"Абсолютна\nпохибка\n(аргумент)", u"Абсолютна\nпохибка\n(функція)")
    fmt_a, fmt_b, fmt_solution_exact, fmt_target_value = r"%.", r"%.", r"%.", r"%."
    fmt_a += r"%sf" % str(main_digits_num(a, int(-np.log10(epsilon))))
    fmt_b += r"%sf" % str(main_digits_num(b, int(-np.log10(epsilon))))
    fmt_solution_exact += r"%sf" % str(main_digits_num(solution_exact, int(-np.log10(epsilon))))
    fmt_target_value += r"%sf" % str(main_digits_num(f(solution_exact), int(-np.log10(epsilon))))
    x = np.linspace(a - additional_length_multiplier_x * (b - a), b + additional_length_multiplier_x * (b - a), dot_num)
    y = f(x)
    ncols, nrows = len(table_head), methods_number + 1
    for i in range(methods_number):
        a_lst.append([])
        b_lst.append([])
        solution_numerical = optimization_methods[i](f, a, b, a_lst[i], b_lst[i], target=target, epsilon=epsilon, iter_lim=iter_lim)
        plt.figure(num=i + 1, figsize=figsize)
        plt.title(optimization_methods_names[i])
        plt.plot(x, y, plot_style,
                 [a, a], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                          np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style,
                 [b, b], [np.min(y) - additional_length_multiplier_y * (np.max(y) - np.min(y)),
                          np.max(y) + additional_length_multiplier_y * (np.max(y) - np.min(y))], borders_style)
        # plt.plot(np.array(a_lst[i]), np.zeros_like(a_lst[i]) + np.min(y) - additional_length_multiplier_y *
        #          (np.max(y) - np.min(y)), a_style, alpha=0.5, label=r'$a_k$')
        # plt.plot(np.array(b_lst[i]), np.zeros_like(b_lst[i]) + np.min(y) - additional_length_multiplier_y *
        #          (np.max(y) - np.min(y)), b_style, alpha=0.5, label=r'$b_k$')
        plt.plot(solution_numerical, f(solution_numerical), solution_style, label=u'Чисельний розв\'язок')
        plt.legend(loc='best', shadow=False, fontsize='x-large', facecolor='black', framealpha=0.15)
        plt.grid(True)
        data.append([optimization_methods_names[i], solution_numerical, f(solution_numerical),
                     np.abs(solution_numerical - solution_exact), np.abs(f(solution_numerical) - f(solution_exact))])
    plt.figure(num=methods_number + 1, figsize=figsize)
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
    plt.title(r"$Задача: f(x) = \frac{x}{x^2 + 1} \rightarrow %s, x\in[$" % target + fmt_a % a + r"; " + fmt_b % b
              + r"]%sТочное решение:" % "\n\n" + r"$x_{%s}$ = " % target + fmt_solution_exact % solution_exact +
              r"; $f(x_{%s})$ = " % target + fmt_target_value % f(solution_exact))
    plt.show()
    plt.close()
    print("a:")
    for arr in a_lst:
        for a in arr:
            print("%.4f" % a)
        print("---------------")
    print("b:")
    for arr in b_lst:
        for b in arr:
            print("%.4f" % b)
        print("---------------")
    print("f(a):")
    for arr in a_lst:
        for a in arr:
            print("%.4f" % f(a))
        print("---------------")
    print("f(b):")
    for arr in b_lst:
        for b in arr:
            print("%.4f" % f(b))
        print("---------------")


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
