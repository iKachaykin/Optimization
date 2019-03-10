import numpy as np
from scipy.integrate import dblquad

partition_number, product_number = 9, 3

tau = np.array([
    [2.99998902, 2.99989407, 1.09085035, 2.8792339, 2.99998692, 2.99992761, 2.42335301, 3.00000557, 2.99999821],
    [7.49999618, 2.49996248, 14.65994657, 17.97359018, 15.79999583, 15.80003266, 16.25610292, 2.50001267, 15.80001706]
])

psi = np.array([-5.23963666e-05, 4.64792331e-05, 6.59965374e-05, 6.59965374e-05, 9.23483129e-05, 2.06068686e-04,
                6.59965374e-05, 1.20476204e-04, 4.86125998e-09])

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
