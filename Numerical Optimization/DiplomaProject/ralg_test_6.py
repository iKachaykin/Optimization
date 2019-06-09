import numpy as np
import matplotlib.pyplot as plt
import NonLinearOptimization as nlopt
from numpy.linalg import norm
from time import time


# Данный файл является шаблоном для использования реализованного r-алгоритма

# Целевая функция
# ----------------------------------------------------------------------------------------------------------------------
# Аргументы:
# x - вектор переменных задачи оптимизации
# args - tuple дополнительных параметров целевой функции, которые на момент оптимизации считаются постоянными
def f(x, args):
    return (x[0] + 10.0 * x[1]) ** 2 + 5.0 * (x[2] - x[3]) ** 4 + (x[1] - 2.0 * x[2]) ** 2 + 10.0 * (x[0] - x[3]) ** 4


if __name__ == '__main__':
    # exact_solution - точное решение задачи оптимизации (если неизвестно - присвоить None)
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
    # return_grads - переменная, определяющая возвращать список всех субградиентов или нет
    # step_method - метод поиска шагового множителя; здесь выставлен в 'adaptive'; параметры представлены именно для
    # этого метода
    # default_step - начальный (пробный) шаг
    # step_red_mult - коэффициент уменьшения шага
    # step_incr_mult - коэффициент увеличения шага
    # lim_num - предельное количество шагов, сделанных в одном направлении без увеличения шагового множителя
    # reduction_epsilon - число, заменяющее ноль в проверке критерия остановки
    # цель параметра - избежать погрешностей в случае равенства функции в старой и новой точках
    # points - список точек, полученный в процессе последовательных приближений при помощи r-алгоритма
    # grads - список субградиентов, вычисленных в точках points
    # numerical_solution - численное решение поставленной задачи оптимизации
    exact_solution = np.array([0.0, 0.0, 0.0, 0.0])
    x0 = np.array([10.0, 10.0, 10.0, -10.0])
    args = 0.0
    grad = nlopt.middle_grad_non_matrix_pool
    form = 'B'
    beta = 0.5
    target = 'min'
    grad_epsilon = 1e-8
    calc_epsilon_x = 1e-7
    calc_epsilon_grad = 1e-6
    step_epsilon = 1e-15
    iter_lim = 1000
    return_grads = True
    step_method = 'adaptive'
    default_step = 1.0
    step_red_mult = 0.8
    step_incr_mult = 1.2
    lim_num = 3
    reduction_epsilon = 1e-15
    calc_time = time()
    points, grads = nlopt.r_algorithm(f, x0, args, grad, form, beta, target, grad_epsilon, calc_epsilon_x,
                                      calc_epsilon_grad, step_epsilon, iter_lim, return_grads, step_method=step_method,
                                      default_step=default_step, step_red_mult=step_red_mult,
                                      step_incr_mult=step_incr_mult, lim_num=lim_num,
                                      reduction_epsilon=reduction_epsilon)
    calc_time = time() - calc_time
    print('Время вычислений: %f' % calc_time)
    numerical_solution = points[-1]
    print('Количество итераций: %d' % points.shape[0])
    if exact_solution is not None:
        print('Отклонение от точного решения: %f' % np.linalg.norm(numerical_solution - exact_solution))
    print('Приближения к решению на каждой итерации:\n{0}'.format(points))
    print('Значения целевой функции на каждой итерации:\n')
    for xi in points:
        print(f(xi, args))
    # Если целевая функция является функцией двух переменных - построим линии уровня
    if numerical_solution.size == 2:
        # x_min, x_max, y_min, y_max - параметры определяющие прямоугольник, в котором будут построены линии уровня:
        # будет построен прямоугольник [x_min, x_max] х [y_min, y_max]
        # dot_num - количество точек для построения линий уровня; чем больше данный параметр, тем точнее строятся линии,
        # и тем дольше работает программа
        # figsize - общий размер (вместе с рамкой) фигуры для построения
        # z - уровни, к которым строятся линии уровня (заполняются автоматически,
        # однако можно определить некоторые начальные значения)
        # level_max_diff - параметр, определяющий максимальное расстояние между уровнями;
        # чем данный параметр ближе к нулю, тем больше линий уровня будет нарисовано
        # point_seq_style, way_style, exact_solution_style - стили последовательных приближений, пути и точного решения
        # grid_alpha - коэффициент прозрачности сетки
        # min_dist_between_points - для того, чтобы исключить построение точек, что очень близки друг к другу, и
        # визуально неразличимы, удалим из points точки, расстояние между которыми до своих предыдущих меньше,
        # чем данный параметр; также пересчитаем заново субградиенты в этих точках
        # grads_color - цвет градиентов
        # plot_grads, label_levels - флаги, нужно ли строить градиенты, подписывать линии уровня своими значениями
        # соответственно
        # points_seq - копия points, из которой исключили достаточно близкие друг к другу точки
        # grads_seq - градиенты, вычисленные в points_seq
        # number_of_plotting_grads - количество градиентов, которое будет построено, если plot_grads=True
        shift_left, shift_right, shift_bottom, shift_top = 5.0, 5.0, 5.0, 5.0
        if exact_solution is not None:
            x_min, x_max, y_min, y_max = \
                exact_solution[0] - shift_left, exact_solution[0] + shift_right,\
                exact_solution[1] - shift_bottom, exact_solution[1] + shift_top
        else:
            x_min, x_max, y_min, y_max = \
                numerical_solution[0] - shift_left, numerical_solution[0] + shift_right,\
                numerical_solution[1] - shift_bottom, numerical_solution[1] + shift_top
        dot_num = 1000
        figsize = (15, 7.5)
        levels = []
        level_max_diff = 10
        point_seq_style, way_style, exact_solution_style = "ko", "k-", "ro"
        grid_alpha = 0.25
        min_dist_between_points = 1e-2
        grads_color = "r"
        plot_grads, label_levels = True, True
        number_of_plotting_grads = 3
        points_seq = nlopt.remove_nearly_same_points(points, min_dist_between_points)
        grads_seq = []
        for point in points_seq:
            grads_seq.append(grad(point, lambda x: f(x, args), grad_epsilon))
        grads_seq = np.array(grads_seq)
        levels = np.sort(f([points_seq[:, 0], points_seq[:, 1]], args))
        count = 0
        while count < levels.size - 1:
            if levels[count + 1] - levels[count] > level_max_diff:
                levels = np.insert(levels, count + 1, (levels[count + 1] + levels[count]) / 2.0)
                count -= 1
            count += 1
        levels = np.array(list(set(levels)))
        levels.sort()
        x, y = np.linspace(x_min, x_max, dot_num), np.linspace(y_min, y_max, dot_num)
        xx, yy = np.meshgrid(x, y, sparse=True)
        z = f([xx, yy], args)
        plt.figure(figsize=figsize)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.grid(True)
        numerical_contour = plt.contour(x, y, z, levels=levels)
        plt.plot(points_seq[:, 0], points_seq[:, 1], point_seq_style, label=u"Наближення")
        for i in range(points_seq.shape[0] - 1):
            plt.plot([points_seq[i][0], points_seq[i + 1][0]], [points_seq[i][1], points_seq[i + 1][1]], way_style)
        if plot_grads:
            for i in range(number_of_plotting_grads):
                grad_coords = -grads_seq[i]
                if norm(grad_coords) > 1:
                    grad_coords = grad_coords / norm(grad_coords) / 2
                plt.arrow(points_seq[i, 0], points_seq[i, 1], grad_coords[0], grad_coords[1], color=grads_color,
                          head_width=0.1, head_length=0.1)
        if label_levels:
            plt.clabel(numerical_contour, inline=1, fontsize=10)
        if exact_solution is not None:
            plt.plot(exact_solution[0], exact_solution[1], exact_solution_style, label=u"Розв'язок")
        plt.legend(loc="best")
        plt.show()
        plt.close()
