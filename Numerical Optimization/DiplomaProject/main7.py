import numpy as np
import numpy.linalg as linalg
from numpy.random import rand
from tqdm import tqdm
from NonLinearOptimization import r_algorithm
import psycopg2


def f(x, args):
    a, b, c = args
    if x.shape[0] == 2:
        return 1 / 2 * (a[0, 0] * x[0] ** 2 + (a[0, 1] + a[1, 0]) * x[0] * x[1] + a[1, 1] * x[1] ** 2) - \
               b[0] * x[0] - b[1] * x[1] + c
    if len(x.shape) == 2:
        return 1 / 2 * np.diagonal(np.dot(np.dot(a, x).T, x)) - np.dot(b, x) + c * np.ones(x.shape[1])
    else:
        return 1 / 2 * np.dot(np.dot(a, x), x) - np.dot(b, x) + c


def grad_f(x, f, eps, args):
    a, b, c = args
    return 1 / 2 * np.dot(a + a.T, x) - b


def main():
    conn = psycopg2.connect('dbname=optimization user=postgres password=postgres_ivan')
    cur = conn.cursor()
    dimension_number, exp_number = 5, 10000
    avg, result_satisf_num = 0.0, 0
    not_so_bad_eps, very_bad_eps, very_bad_a, very_bad_b, very_bad_x_exact, very_bad_x_numerical = \
        [], [], [], [], [], []
    square_delta = 10
    x0, uniform_distr_low, uniform_distr_high, calc_epsilon = \
        np.zeros(dimension_number), -5, 5, 1e-6
    a, b, c = \
        rand(dimension_number, dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low, \
        rand(dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low, 0

    abs_err, rel_err, iters = np.zeros(exp_number), np.zeros(exp_number), np.zeros(exp_number)

    tqdm.monitor_interval = 0
    for j in tqdm(range(exp_number)):
        b = rand(dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low
        while True:
            a = rand(dimension_number, dimension_number) * (uniform_distr_high - uniform_distr_low) + uniform_distr_low
            hessian_of_f = (a + a.T) / 2
            flag = False
            for i in range(dimension_number):
                minor = linalg.det(hessian_of_f[:i+1, :i+1])
                if minor < 1e-15:
                    flag = True
                    break
            if not flag:
                break
        exact_solution = linalg.solve(hessian_of_f, b)
        x0 = rand(dimension_number) * 2 * square_delta + (x0 - square_delta)
        points_seq = r_algorithm(f, x0, args=(a, b, c), grad=lambda x, f, epsilon: grad_f(x, f, epsilon, (a, b, c)), form='B',
                                 calc_epsilon_x=calc_epsilon, iter_lim=1000, step_method='adaptive', default_step=1.0,
                                 step_red_mult=0.9, step_incr_mult=1.1, lim_num=3, reduction_epsilon=1e-15)
        argmin = points_seq[points_seq.shape[0] - 1]
        abs_err[j] = linalg.norm(exact_solution - argmin)
        rel_err[j] = error = linalg.norm(exact_solution - argmin) / linalg.norm(argmin)
        iters[j] = points_seq.shape[0]
        norm_exact = linalg.norm(exact_solution)
        norm_numerical = linalg.norm(argmin)
        if error < 1000 * calc_epsilon:
            result_satisf_num += 1
            avg += (points_seq.shape[0] - 1) / exp_number
            point_class = 0
        elif error < 1:
            not_so_bad_eps.append(linalg.norm(exact_solution - argmin))
            point_class = 1
        else:
            very_bad_eps.append(linalg.norm(exact_solution - argmin))
            very_bad_a.append(a)
            very_bad_b.append(b)
            very_bad_x_exact.append(exact_solution)
            very_bad_x_numerical.append(argmin)
            point_class = 2
        cur.execute('insert into numerical_results(eps, det, iter, point_class, norm_exact, norm_numerical) '
                    'values (%.15f, %.15f, %d, %d, %.15f, %.15f)' %
                    (error, minor, points_seq.shape[0] - 1, point_class, norm_exact, norm_numerical))
    print('abs_err:\navg: %.16f\nstd: %.16f\nmax: %.16f\nmedian: %.16f' %
          (abs_err.mean(), abs_err.std(), abs_err.max(), float(np.median(abs_err))))
    print('rel_err:\navg: %.16f\nstd: %.16f\nmax: %.16f\nmedian: %.16f' %
          (rel_err.mean(), rel_err.std(), rel_err.max(), float(np.median(rel_err))))
    print('iters:\navg: %f\nstd: %f\nmax: %d\nmedian: %.16f' %
          (iters.mean(), iters.std(), iters.max(), float(np.median(iters))))
    not_so_bad_eps = np.array(not_so_bad_eps)
    avg = avg * (exp_number / result_satisf_num)
    print('Number of absolutely success cases: %d' % result_satisf_num)
    print('Success in %.2f%% of cases' % (result_satisf_num / exp_number * 100))
    print('Average number of iterations in case of convergence: %d-%d' %
          (np.floor(avg), np.ceil(avg)))
    print('Number of not so bad cases: %d' % len(not_so_bad_eps))
    print('Not so bad in %.2f%% of cases' % (len(not_so_bad_eps) / exp_number * 100))
    if len(not_so_bad_eps) > 0:
        print('Average error for not so bad error: %f' % not_so_bad_eps.mean())
        print('Average relation between "not so bad error" and allowable error: %f' %
              (not_so_bad_eps.mean() / calc_epsilon))
    print('Number of bad cases: %d' % len(very_bad_eps))
    print('Bad in %.2f%% of cases' % (len(very_bad_eps) / exp_number * 100))
    for v_eps, v_a, v_b, v_x_exact, v_x_num in \
            zip(very_bad_eps, very_bad_a, very_bad_b, very_bad_x_exact, very_bad_x_numerical):
        print('----------------------------------------------------------------')
        print('Very bad eps is: %f' % v_eps)
        print('Very bad A:')
        print(v_a)
        print('Very bad B:')
        print(v_b)
        print('Very bad exact solution:')
        print(v_x_exact)
        print('Very bad numerical solution:')
        print(v_x_num)
        v_hess = (v_a + v_a.T) / 2
        for i in range(dimension_number):
            print('Corner\'s minor #%d is %f' %
                  (i+1, linalg.det(v_hess[:i+1, :i+1])))
    conn.commit()
    conn.close()
    cur.close()


if __name__ == '__main__':
    main()
