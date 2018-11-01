import numpy as np
import psycopg2
from NonLinearOptimization import r_algorithm, middle_grad_non_matrix, gsection
from tqdm import tqdm


def get_x_sample_from_sample(sample, start, end):
    x_sample = []
    for part in sample:
        x_sample.append(np.array(part[start:end]))
    return np.array(x_sample)


def get_y_sample_from_sample(sample, y_index):
    y_sample = []
    for part in sample:
        y_sample.append(part[y_index])
    return np.array(y_sample)


def get_x_linguistic_sample_from_sample(sample, row_start, row_end, col_start, col_end):
    return get_x_sample_from_sample(sample[row_start:row_end], col_start, col_end)

def mu_partial_b(x, b, c):
    return 2 * c ** 2 * (x - b) / (c * c + (x - b) ** 2) ** 2

def mu_partial_c(x, b, c):
    return 2 * c * (x - b) ** 2 / (c * c + (x - b) ** 2) ** 2

def mu_class_partial_derivative_weight(weights, weight_index, x, b, c, class_index, class_number, sample_with_classes):
    w_starts, w_distr = weights_starts(class_number), weights_by_class_distribution(class_number)
    if weight_index < w_starts[class_index] or weight_index >= w_starts[class_index] + w_distr[class_index]:
        return 0
    if mu_y_class(x, b, c, weights, w_starts[class_index], w_starts[class_index] + w_distr[class_index],
                  sample_with_classes) >= 1 - 1e-10:
        return 0
    comp = 1
    b_matrix, c_matrix = b.copy().reshape(x_sample.shape[1], 3), c.copy().reshape(x_sample.shape[1], 3)
    x_ling_sample = get_x_linguistic_sample_from_sample(sample_with_classes, 0, 24, 2, 8)
    for i in range(len(x)):
        if x_ling_sample[weight_index, i] == 'В':
            bi = b_matrix[i, 0]
            ci = c_matrix[i, 0]
        elif x_ling_sample[weight_index, i] == 'С':
            bi = b_matrix[i, 1]
            ci = c_matrix[i, 1]
        else:
            bi = b_matrix[i, 2]
            ci = c_matrix[i, 2]
        comp *= mu(x[i], bi, ci)
    return comp


def mu_class_partial_derivative_b(b, b_index_row, b_index_col, x, c, weights, class_index, class_number, sample_with_classes):
    w_starts, w_distr = weights_starts(class_number), weights_by_class_distribution(class_number)
    if mu_y_class(x, b, c, weights, w_starts[class_index], w_starts[class_index] + w_distr[class_index],
                  sample_with_classes) >= 1 - 1e-10:
        return 0
    class_indexes = {0: 'В', 1: 'Н', 2: 'С'}
    b_matrix, c_matrix = b.copy().reshape(x_sample.shape[1], 3), c.copy().reshape(x_sample.shape[1], 3)
    sum = 0
    x_ling_sample = get_x_linguistic_sample_from_sample(sample_with_classes, w_starts[class_index],
                                                        w_starts[class_index] + w_distr[class_index], 2, 8)
    for w, x_ling in zip(weights[w_starts[class_index]:w_starts[class_index] + w_distr[class_index]], x_ling_sample):
        comp = w
        for i in range(len(x)):
            if i == b_index_row:
                if x_ling[i] == class_indexes.get(b_index_col):
                    comp *= mu_partial_b(x[i], b_matrix[b_index_row, b_index_col], c_matrix[b_index_row, b_index_col])
                else:
                    comp *= 0
                continue
            if x_ling[i] == 'В':
                bi = b_matrix[i, 0]
                ci = c_matrix[i, 0]
            elif x_ling[i] == 'С':
                bi = b_matrix[i, 1]
                ci = c_matrix[i, 1]
            else:
                bi = b_matrix[i, 2]
                ci = c_matrix[i, 2]
            comp *= mu(x[i], bi, ci)
        sum += comp
    return sum

def mu_class_partial_derivative_c(c, c_index_row, c_index_col, x, b, weights, class_index, class_number, sample_with_classes):
    w_starts, w_distr = weights_starts(class_number), weights_by_class_distribution(class_number)
    if mu_y_class(x, b, c, weights, w_starts[class_index], w_starts[class_index] + w_distr[class_index],
                  sample_with_classes) >= 1 - 1e-10:
        return 0
    class_indexes = {0: 'В', 1: 'Н', 2: 'С'}
    b_matrix, c_matrix = b.copy().reshape(x_sample.shape[1], 3), c.copy().reshape(x_sample.shape[1], 3)
    sum = 0
    x_ling_sample = get_x_linguistic_sample_from_sample(sample_with_classes, w_starts[class_index],
                                                        w_starts[class_index] + w_distr[class_index], 2, 8)
    for w, x_ling in zip(weights[w_starts[class_index]:w_starts[class_index] + w_distr[class_index]], x_ling_sample):
        comp = w
        for i in range(len(x)):
            if i == c_index_row:
                if x_ling[i] == class_indexes.get(c_index_col):
                    comp *= mu_partial_c(x[i], b_matrix[c_index_row, c_index_col], c_matrix[c_index_row, c_index_col])
                else:
                    comp *= 0
                continue
            if x_ling[i] == 'В':
                bi = b_matrix[i, 0]
                ci = c_matrix[i, 0]
            elif x_ling[i] == 'С':
                bi = b_matrix[i, 1]
                ci = c_matrix[i, 1]
            else:
                bi = b_matrix[i, 2]
                ci = c_matrix[i, 2]
            comp *= mu(x[i], bi, ci)
        sum += comp
    return sum


def y_partial_weight(weights, weights_index, x, b, c, class_number, sample_with_classes, class_values):
    sum_1, sum_2, sum_3, sum_4 = 0, 0, 0, 0
    w_starts, w_distr = weights_starts(class_number), weights_by_class_distribution(class_number)
    for i in range(class_number):
        sum_1 += class_values[i] * mu_class_partial_derivative_weight(weights, weights_index, x, b, c, i, class_number,
                                                                      sample_with_classes)
        sum_2 += class_values[i] * mu_y_class(x, b, c, weights, w_starts[i], w_starts[i] + w_distr[i], sample_with_classes)
        sum_3 += mu_class_partial_derivative_weight(weights, weights_index, x, b, c, i, class_number, sample_with_classes)
        sum_4 += mu_y_class(x, b, c, weights, w_starts[i], w_starts[i] + w_distr[i], sample_with_classes)
    return (sum_1 * sum_4 - sum_2 * sum_3) / sum_4 / sum_4


def y_partial_b(b, b_index_row, b_index_col, x, c, weights, class_number, sample_with_classes, class_values):
    sum_1, sum_2, sum_3, sum_4 = 0, 0, 0, 0
    w_starts, w_distr = weights_starts(class_number), weights_by_class_distribution(class_number)
    for i in range(class_number):
        sum_1 += class_values[i] * mu_class_partial_derivative_b(b, b_index_row, b_index_col, x, c, weights, i,
                                                                 class_number, sample_with_classes)
        sum_2 += class_values[i] * mu_y_class(x, b, c, weights, w_starts[i], w_starts[i] + w_distr[i], sample_with_classes)
        sum_3 += mu_class_partial_derivative_b(b, b_index_row, b_index_col, x, c, weights, i, class_number,
                                               sample_with_classes)
        sum_4 += mu_y_class(x, b, c, weights, w_starts[i], w_starts[i] + w_distr[i], sample_with_classes)
    return (sum_1 * sum_4 - sum_2 * sum_3) / sum_4 / sum_4


def y_partial_c(c, c_index_row, c_index_col, x, b, weights, class_number, sample_with_classes, class_values):
    sum_1, sum_2, sum_3, sum_4 = 0, 0, 0, 0
    w_starts, w_distr = weights_starts(class_number), weights_by_class_distribution(class_number)
    for i in range(class_number):
        sum_1 += class_values[i] * mu_class_partial_derivative_c(c, c_index_row, c_index_col, x, b, weights, i,
                                                                 class_number, sample_with_classes)
        sum_2 += class_values[i] * mu_y_class(x, b, c, weights, w_starts[i], w_starts[i] + w_distr[i], sample_with_classes)
        sum_3 += mu_class_partial_derivative_c(c, c_index_row, c_index_col, x, b, weights, i, class_number,
                                               sample_with_classes)
        sum_4 += mu_y_class(x, b, c, weights, w_starts[i], w_starts[i] + w_distr[i], sample_with_classes)
    return (sum_1 * sum_4 - sum_2 * sum_3) / sum_4 / sum_4

def target_partial_weight(x_sample, y_sample, weights, weights_index, b, c, class_number, sample_with_classes,
                          classes_values):
    sum_1, sum_2 = 0, 0
    for i in range(len(y_sample)):
        sum_1 += (mu_y(x_sample[i], b, c, weights, class_number, classes_values, sample_with_classes) - y_sample[i]) * \
                 y_partial_weight(weights, weights_index, x_sample[i], b, c, class_number, sample_with_classes,
                                  classes_values)
        sum_2 += (mu_y(x_sample[i], b, c, weights, class_number, classes_values, sample_with_classes) - y_sample[i])**2
    return sum_1 / np.sqrt(sum_2)

def target_partial_b(x_sample, y_sample, weights, b, b_index_row, b_index_col, c, class_number, sample_with_classes,
                          classes_values):
    sum_1, sum_2 = 0, 0
    for i in range(len(y_sample)):
        sum_1 += (mu_y(x_sample[i], b, c, weights, class_number, classes_values, sample_with_classes) - y_sample[i]) * \
                 y_partial_b(b, b_index_row, b_index_col, x_sample[i], c, weights, class_number, sample_with_classes,
                             classes_values)
        sum_2 += (mu_y(x_sample[i], b, c, weights, class_number, classes_values, sample_with_classes) - y_sample[
            i]) ** 2
    return sum_1 / np.sqrt(sum_2)

def target_partial_c(x_sample, y_sample, weights, b, c, c_index_row, c_index_col, class_number, sample_with_classes,
                          classes_values):
    sum_1, sum_2 = 0, 0
    for i in range(len(y_sample)):
        sum_1 += (mu_y(x_sample[i], b, c, weights, class_number, classes_values, sample_with_classes) - y_sample[i]) * \
                 y_partial_c(c, c_index_row, c_index_col, x_sample[i], b, weights, class_number, sample_with_classes,
                             classes_values)
        sum_2 += (mu_y(x_sample[i], b, c, weights, class_number, classes_values, sample_with_classes) - y_sample[
            i]) ** 2
    return sum_1 / np.sqrt(sum_2)


def target_grad(target_vector, target, epsilon=10e-6):
    weights = target_vector[weight_start:weight_end]
    b = target_vector[b_start:b_end]
    c = target_vector[c_start:c_end]
    res = []
    tqdm.monitor_interval = 0
    for i in tqdm(range(len(weights))):
        res.append(target_partial_weight(x_sample, y_sample, weights, i, b, c, class_number, sample_with_classes,
                                         class_values))
    for i in tqdm(range(variable_number)):
        for j in range(class_number):
            res.append(target_partial_b(x_sample, y_sample, weights, b, i, j, c, class_number, sample_with_classes,
                                        class_values))

    for i in tqdm(range(variable_number)):
        for j in range(class_number):
            res.append(target_partial_c(x_sample, y_sample, weights, b, c, i, j, class_number, sample_with_classes,
                                        class_values))
    return np.array(res)

def weights_starts(class_number):
    conn = psycopg2.connect('dbname=optimization user=postgres password=postgres_ivan')
    cur = conn.cursor()
    result = []
    for cn in range(class_number):
        cur.execute("select elem_id from sample_with_classes where y_class = 'D%d' group by elem_id" %
                    (cn + 1))
        result.append(cur.fetchall()[0][0] - 1)
    cur.close()
    conn.close()
    return result


def weights_by_class_distribution(class_number):
    conn = psycopg2.connect('dbname=optimization user=postgres password=postgres_ivan')
    cur = conn.cursor()
    result = []
    for cn in range(class_number):
        cur.execute("select count(y_class) from sample_with_classes where y_class = 'D%d'" %
                    (cn + 1))
        result.append(cur.fetchall()[0][0])
    cur.close()
    conn.close()
    return result


def mu(x, b, c):
    return 1 / (1 + ((x - b) / c) ** 2)


def mu_y_class(x, b, c, weights, weights_start, weights_end, sample_with_classes):
    sum = 0.0
    b_matrix, c_matrix = b.copy().reshape(x_sample.shape[1], 3), c.copy().reshape(x_sample.shape[1], 3)
    x_ling_sample = get_x_linguistic_sample_from_sample(sample_with_classes, weights_start, weights_end, 2, 8)
    for w, x_ling in zip(weights[weights_start:weights_end], x_ling_sample):
        comp = w
        for i in range(len(x)):
            if x_ling[i] == 'В':
                bi = b_matrix[i, 0]
                ci = c_matrix[i, 0]
            elif x_ling[i] == 'С':
                bi = b_matrix[i, 1]
                ci = c_matrix[i, 1]
            else:
                bi = b_matrix[i, 2]
                ci = c_matrix[i, 2]
            comp *= mu(x[i], bi, ci)
        sum += comp
    return np.minimum(1, sum)


def mu_y(x, b, c, weights, class_number, classes_values, sample_with_classes):
    w_starts, w_distr = weights_starts(class_number), weights_by_class_distribution(class_number)
    mu_y_class_results = []
    for i, size in zip(w_starts, w_distr):
        mu_y_class_results.append(mu_y_class(x, b, c, weights, i, i + size, sample_with_classes))
    mu_y_class_results = np.array(mu_y_class_results)
    return np.dot(classes_values, mu_y_class_results) / np.sum(mu_y_class_results)


def y_error(x_sample, y_sample, b, c, weights, class_number, classes_values, sample_with_classes):
    y = np.zeros_like(y_sample)
    for i in range(y_sample.size):
        y[i] = mu_y(x_sample[i], b, c, weights, class_number, classes_values, sample_with_classes)
    return np.sqrt(((y - y_sample) ** 2).sum())


def target_function(target_vector, args):
    weight_start, weight_end, b_start, b_end, c_start, c_end, x_sample, y_sample, class_number, class_values,\
        sample_with_classes = args
    weights = target_vector[weight_start:weight_end]
    b = target_vector[b_start:b_end]
    c = target_vector[c_start:c_end]
    return y_error(x_sample, y_sample, b, c, weights, class_number, class_values, sample_with_classes)


if __name__ == '__main__':
    variable_number, class_number, class_values = 6, 3, np.array([0, 5, 10])
    conn = psycopg2.connect('dbname=optimization user=postgres password=postgres_ivan')
    cur = conn.cursor()
    cur.execute('select * from sample_with_classes')
    sample_with_classes = cur.fetchall()
    cur.execute('select * from sample')
    sample = cur.fetchall()
    x_sample = get_x_sample_from_sample(sample, 2, 8)
    y_sample = get_y_sample_from_sample(sample, 8)
    weight_vector = np.ones(len(sample))
    b_vector, c_vector = np.zeros(variable_number * class_number), np.zeros(variable_number * class_number) + 2.5
    weight_vector_dim, b_vector_dim, c_vector_dim = \
        len(sample), variable_number * class_number, variable_number * class_number
    for i in range(len(b_vector)):
        if i % 3 == 1:
            b_vector[i] = 5
        if i % 3 == 2:
            b_vector[i] = 10
    target_vector = np.append(weight_vector, np.append(b_vector, c_vector))
    print(target_vector)
    weight_start, weight_end = 0, weight_vector.size
    b_start, b_end = weight_vector.size, weight_vector.size + b_vector.size
    c_start, c_end = weight_vector.size + b_vector.size, weight_vector.size + b_vector.size + c_vector.size
    #print(y_error(x_sample, y_sample, b_vector, c_vector, weight_vector, class_number, class_values))
    print(target_function(target_vector, args=(weight_start, weight_end, b_start, b_end, c_start, c_end,
                                               x_sample, y_sample, class_number, class_values, sample_with_classes)))
    print('-------------------------------')
    # for x in x_sample:
    #     tmp_b, tmp_c, tmp_weights = \
    #         np.random.rand(variable_number * class_number), \
    #         np.random.rand(variable_number * class_number), \
    #         np.random.rand(len(sample))
        # print(mu_y(x, tmp_b, tmp_c, tmp_weights, class_number, class_values))
    #    print(y_error(x_sample, y_sample, tmp_b, tmp_c, tmp_weights, class_number, class_values))
    for x in x_sample:
        print('%f' % mu_y(x, b_vector, c_vector, weight_vector, class_number, class_values,
                                sample_with_classes))
    solution = r_algorithm(target_function, target_vector, grad=middle_grad_non_matrix,#target_grad,
                           args=(weight_start, weight_end, b_start, b_end, c_start, c_end, x_sample, y_sample,
                                 class_number, class_values, sample_with_classes),
                           form='B', calc_epsilon=1e-2, iter_lim=100, step_method='adaptive', default_step=10,
                           step_red_mult=0.75, step_incr_mult=1.5, lim_num=3, reduction_epsilon=1e-15, grad_epsilon=1)[0]
    # solution = r_algorithm(target_function, target_vector, grad=middle_grad_non_matrix,#target_grad,
    #                        args=(weight_start, weight_end, b_start, b_end, c_start, c_end, x_sample, y_sample,
    #                              class_number, class_values, sample_with_classes),
    #                        form='B', calc_epsilon=1e-2, iter_lim=100, step_method='argmin', step_min=0.0,
    #                        step_max=100.0, argmin_finder=gsection, grad_epsilon=1)[0]
    solution = solution[len(solution) - 1]
    print(solution)
    print(target_function(solution, (weight_start, weight_end, b_start, b_end, c_start, c_end, x_sample, y_sample,
                                     class_number, class_values, sample_with_classes)))
    weight_vector_sol = solution[weight_start:weight_end]
    b_vector_sol = solution[b_start:b_end]
    c_vector_sol = solution[c_start:c_end]
    b_matrix_sol, c_matrix_sol = b_vector_sol.copy().reshape(x_sample.shape[1], 3),\
                                 c_vector_sol.copy().reshape(x_sample.shape[1], 3)
    print('------------------------\nweights\n')
    for w_old, w_new in zip(weight_vector, weight_vector_sol):
        print('%f - %f' % (w_old, w_new))
    print('-------------------------\nb\n')
    print(b_matrix_sol)
    print('-------------------------\nc\n')
    print(c_matrix_sol)
    for x in x_sample:
        print('%f' % mu_y(x, b_vector_sol, c_vector_sol, weight_vector_sol, class_number, class_values,
                                sample_with_classes))
    print('--------------------------\n')
    for y in y_sample:
        print('%f' % y)
    cur.close()
    conn.close()
