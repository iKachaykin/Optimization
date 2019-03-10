import numpy as np
from scipy.integrate import dblquad

partition_number, product_number = 9, 3

tau = np.array([
    [2.99980429, 2.99933857, 0.38583195, 0.09607969, 3.00038762, 2.99941846, 1.61276445, 3.00107942, 2.99991959],
    [3.99996753, 10.39931981, 17.25781821, 10.85520369, 10.46205448, 16.80123233, 12.56414143, 16.80052497, 14.69992265]
])

psi = np.array([-1.08034517e-04, 4.63585225e-05, 5.68664641e-05, 5.68664641e-05, 1.52474531e-05, 2.18857993e-04,
                5.68664641e-05, 1.90335358e-04, 6.90747286e-10])

a_matrix = np.ones((partition_number, product_number)) * 100.0
a_matrix[0, 0] = 0.0
a_matrix[0, 1] = 0.0
a_matrix[0, 2] = 0.0
a_matrix[1, 2] = 0.0
a_matrix[4, 1] = 0.0
a_matrix[5, 2] = 0.0
a_matrix[7, 1] = 0.0
a_matrix[8, 0] = 0.0

b_vector = np.ones(partition_number) * 20.0
b_vector[0] = 200.0
b_vector[1] = 50.0
b_vector[4] = 60.0
b_vector[8] = 120.0


def dual_target(y, x):

    return np.amin([np.abs(x - tau[0, 0]) + np.abs(y - tau[1, 0]) + a_matrix[0, 0] + psi[0],
                    np.abs(x - tau[0, 1]) + np.abs(y - tau[1, 1]) + a_matrix[1, 0] + psi[1],
                    np.abs(x - tau[0, 2]) + np.abs(y - tau[1, 2]) + a_matrix[2, 0] + psi[2],
                    np.abs(x - tau[0, 3]) + np.abs(y - tau[1, 3]) + a_matrix[3, 0] + psi[3],
                    np.abs(x - tau[0, 4]) + np.abs(y - tau[1, 4]) + a_matrix[4, 0] + psi[4],
                    np.abs(x - tau[0, 5]) + np.abs(y - tau[1, 5]) + a_matrix[5, 0] + psi[5],
                    np.abs(x - tau[0, 6]) + np.abs(y - tau[1, 6]) + a_matrix[6, 0] + psi[6],
                    np.abs(x - tau[0, 7]) + np.abs(y - tau[1, 7]) + a_matrix[7, 0] + psi[7],
                    np.abs(x - tau[0, 8]) + np.abs(y - tau[1, 8]) + a_matrix[8, 0] + psi[8]]) + \
           np.amin([np.abs(x - tau[0, 0]) + np.abs(y - tau[1, 0]) + a_matrix[0, 1] + psi[0],
                    np.abs(x - tau[0, 1]) + np.abs(y - tau[1, 1]) + a_matrix[1, 1] + psi[1],
                    np.abs(x - tau[0, 2]) + np.abs(y - tau[1, 2]) + a_matrix[2, 1] + psi[2],
                    np.abs(x - tau[0, 3]) + np.abs(y - tau[1, 3]) + a_matrix[3, 1] + psi[3],
                    np.abs(x - tau[0, 4]) + np.abs(y - tau[1, 4]) + a_matrix[4, 1] + psi[4],
                    np.abs(x - tau[0, 5]) + np.abs(y - tau[1, 5]) + a_matrix[5, 1] + psi[5],
                    np.abs(x - tau[0, 6]) + np.abs(y - tau[1, 6]) + a_matrix[6, 1] + psi[6],
                    np.abs(x - tau[0, 7]) + np.abs(y - tau[1, 7]) + a_matrix[7, 1] + psi[7],
                    np.abs(x - tau[0, 8]) + np.abs(y - tau[1, 8]) + a_matrix[8, 1] + psi[8]]) + \
           np.amin([np.abs(x - tau[0, 0]) + np.abs(y - tau[1, 0]) + a_matrix[0, 2] + psi[0],
                    np.abs(x - tau[0, 1]) + np.abs(y - tau[1, 1]) + a_matrix[1, 2] + psi[1],
                    np.abs(x - tau[0, 2]) + np.abs(y - tau[1, 2]) + a_matrix[2, 2] + psi[2],
                    np.abs(x - tau[0, 3]) + np.abs(y - tau[1, 3]) + a_matrix[3, 2] + psi[3],
                    np.abs(x - tau[0, 4]) + np.abs(y - tau[1, 4]) + a_matrix[4, 2] + psi[4],
                    np.abs(x - tau[0, 5]) + np.abs(y - tau[1, 5]) + a_matrix[5, 2] + psi[5],
                    np.abs(x - tau[0, 6]) + np.abs(y - tau[1, 6]) + a_matrix[6, 2] + psi[6],
                    np.abs(x - tau[0, 7]) + np.abs(y - tau[1, 7]) + a_matrix[7, 2] + psi[7],
                    np.abs(x - tau[0, 8]) + np.abs(y - tau[1, 8]) + a_matrix[8, 2] + psi[8]])


def indicator(y, x, args):
    i, j = args
    for k in range(partition_number):
        if np.sqrt((x-tau[0, i])**2 + (y-tau[1, i])**2) + a_matrix[i, j] + psi[i] > \
                np.sqrt((x-tau[0, k])**2 + (y-tau[1, k])**2) + a_matrix[k, j] + psi[k]:
            return 0.0
    return 1.0


def target(y, x):

    return (np.abs(x - tau[0, 0]) + np.abs(y - tau[1, 0]) + a_matrix[0, 0]) * indicator(y, x, (0, 0)) +\
           (np.abs(x - tau[0, 1]) + np.abs(y - tau[1, 1]) + a_matrix[1, 0]) * indicator(y, x, (1, 0)) +\
           (np.abs(x - tau[0, 2]) + np.abs(y - tau[1, 2]) + a_matrix[2, 0]) * indicator(y, x, (2, 0)) +\
           (np.abs(x - tau[0, 3]) + np.abs(y - tau[1, 3]) + a_matrix[3, 0]) * indicator(y, x, (3, 0)) +\
           (np.abs(x - tau[0, 4]) + np.abs(y - tau[1, 4]) + a_matrix[4, 0]) * indicator(y, x, (4, 0)) +\
           (np.abs(x - tau[0, 5]) + np.abs(y - tau[1, 5]) + a_matrix[5, 0]) * indicator(y, x, (5, 0)) +\
           (np.abs(x - tau[0, 6]) + np.abs(y - tau[1, 6]) + a_matrix[6, 0]) * indicator(y, x, (6, 0)) +\
           (np.abs(x - tau[0, 7]) + np.abs(y - tau[1, 7]) + a_matrix[7, 0]) * indicator(y, x, (7, 0)) +\
           (np.abs(x - tau[0, 8]) + np.abs(y - tau[1, 8]) + a_matrix[8, 0]) * indicator(y, x, (8, 0)) +\
           (np.abs(x - tau[0, 0]) + np.abs(y - tau[1, 0]) + a_matrix[0, 1]) * indicator(y, x, (0, 1)) +\
           (np.abs(x - tau[0, 1]) + np.abs(y - tau[1, 1]) + a_matrix[1, 1]) * indicator(y, x, (1, 1)) +\
           (np.abs(x - tau[0, 2]) + np.abs(y - tau[1, 2]) + a_matrix[2, 1]) * indicator(y, x, (2, 1)) +\
           (np.abs(x - tau[0, 3]) + np.abs(y - tau[1, 3]) + a_matrix[3, 1]) * indicator(y, x, (3, 1)) +\
           (np.abs(x - tau[0, 4]) + np.abs(y - tau[1, 4]) + a_matrix[4, 1]) * indicator(y, x, (4, 1)) +\
           (np.abs(x - tau[0, 5]) + np.abs(y - tau[1, 5]) + a_matrix[5, 1]) * indicator(y, x, (5, 1)) +\
           (np.abs(x - tau[0, 6]) + np.abs(y - tau[1, 6]) + a_matrix[6, 1]) * indicator(y, x, (6, 1)) +\
           (np.abs(x - tau[0, 7]) + np.abs(y - tau[1, 7]) + a_matrix[7, 1]) * indicator(y, x, (7, 1)) +\
           (np.abs(x - tau[0, 8]) + np.abs(y - tau[1, 8]) + a_matrix[8, 1]) * indicator(y, x, (8, 1)) +\
           (np.abs(x - tau[0, 0]) + np.abs(y - tau[1, 0]) + a_matrix[0, 2]) * indicator(y, x, (0, 2)) +\
           (np.abs(x - tau[0, 1]) + np.abs(y - tau[1, 1]) + a_matrix[1, 2]) * indicator(y, x, (1, 2)) +\
           (np.abs(x - tau[0, 2]) + np.abs(y - tau[1, 2]) + a_matrix[2, 2]) * indicator(y, x, (2, 2)) +\
           (np.abs(x - tau[0, 3]) + np.abs(y - tau[1, 3]) + a_matrix[3, 2]) * indicator(y, x, (3, 2)) +\
           (np.abs(x - tau[0, 4]) + np.abs(y - tau[1, 4]) + a_matrix[4, 2]) * indicator(y, x, (4, 2)) +\
           (np.abs(x - tau[0, 5]) + np.abs(y - tau[1, 5]) + a_matrix[5, 2]) * indicator(y, x, (5, 2)) +\
           (np.abs(x - tau[0, 6]) + np.abs(y - tau[1, 6]) + a_matrix[6, 2]) * indicator(y, x, (6, 2)) +\
           (np.abs(x - tau[0, 7]) + np.abs(y - tau[1, 7]) + a_matrix[7, 2]) * indicator(y, x, (7, 2)) +\
           (np.abs(x - tau[0, 8]) + np.abs(y - tau[1, 8]) + a_matrix[8, 2]) * indicator(y, x, (8, 2))


if __name__ == '__main__':

    I = dblquad(dual_target, 0.0, 6.0, lambda x: 0.0, lambda x: 20.0)[0] -\
        np.dot(psi, b_vector)
    print(I)
    I2 = dblquad(target, 0.0, 6.0, lambda x: 0.0, lambda x: 20.0)[0]
    print(I2)
