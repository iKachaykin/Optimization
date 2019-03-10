import numpy as np
from scipy.integrate import dblquad

partition_number = 9

tau = np.array([
    [3.00001749, 0.75079231, 1.0392672, 0.3462155, 2.99997755, 1.54253991, 0.30125642, 2.99999624, 2.9999899],
    [4.22096355, 2.21251011, 0.47196582, 16.3754584, 10.52497296, 8.87571846, 5.43387606, 16.83762889, 14.73760651]
])

psi = np.array([0.0014, 0.0, 0.0, 0.0, 0.0015, 0.0, 0.0, 0.0023, 0.0031])

a_matrix = np.array([
    [0.0, 0.0],
    [100.0, 100.0],
    [100.0, 100.0],
    [100.0, 100.0],
    [100.0, 0.0],
    [100.0, 100.0],
    [100.0, 100.0],
    [100.0, 0.0],
    [0.0, 100.0]
])

b_vector = np.array([150.0, 20.0, 20.0, 20.0, 60.0, 20.0, 20.0, 20.0, 38.0])


def dual_target(y, x):

    return np.amin([np.sqrt((x - tau[0, 0]) ** 2 + (y - tau[1, 0]) ** 2) + a_matrix[0, 0] + psi[0],
                    np.sqrt((x - tau[0, 1]) ** 2 + (y - tau[1, 1]) ** 2) + a_matrix[1, 0] + psi[1],
                    np.sqrt((x - tau[0, 2]) ** 2 + (y - tau[1, 2]) ** 2) + a_matrix[2, 0] + psi[2],
                    np.sqrt((x - tau[0, 3]) ** 2 + (y - tau[1, 3]) ** 2) + a_matrix[3, 0] + psi[3],
                    np.sqrt((x - tau[0, 4]) ** 2 + (y - tau[1, 4]) ** 2) + a_matrix[4, 0] + psi[4],
                    np.sqrt((x - tau[0, 5]) ** 2 + (y - tau[1, 5]) ** 2) + a_matrix[5, 0] + psi[5],
                    np.sqrt((x - tau[0, 6]) ** 2 + (y - tau[1, 6]) ** 2) + a_matrix[6, 0] + psi[6],
                    np.sqrt((x - tau[0, 7]) ** 2 + (y - tau[1, 7]) ** 2) + a_matrix[7, 0] + psi[7],
                    np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 0] + psi[8]]) + \
           np.amin([np.sqrt((x - tau[0, 0]) ** 2 + (y - tau[1, 0]) ** 2) + a_matrix[0, 1] + psi[0],
                    np.sqrt((x - tau[0, 1]) ** 2 + (y - tau[1, 1]) ** 2) + a_matrix[1, 1] + psi[1],
                    np.sqrt((x - tau[0, 2]) ** 2 + (y - tau[1, 2]) ** 2) + a_matrix[2, 1] + psi[2],
                    np.sqrt((x - tau[0, 3]) ** 2 + (y - tau[1, 3]) ** 2) + a_matrix[3, 1] + psi[3],
                    np.sqrt((x - tau[0, 4]) ** 2 + (y - tau[1, 4]) ** 2) + a_matrix[4, 1] + psi[4],
                    np.sqrt((x - tau[0, 5]) ** 2 + (y - tau[1, 5]) ** 2) + a_matrix[5, 1] + psi[5],
                    np.sqrt((x - tau[0, 6]) ** 2 + (y - tau[1, 6]) ** 2) + a_matrix[6, 1] + psi[6],
                    np.sqrt((x - tau[0, 7]) ** 2 + (y - tau[1, 7]) ** 2) + a_matrix[7, 1] + psi[7],
                    np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 1] + psi[8]])


def indicator(y, x, args):
    i, j = args
    for k in range(partition_number):
        if np.sqrt((x-tau[0, i])**2 + (y-tau[1, i])**2) + a_matrix[i, j] + psi[i] > \
                np.sqrt((x-tau[0, k])**2 + (y-tau[1, k])**2) + a_matrix[k, j] + psi[k]:
            return 0.0
    return 1.0


def target(y, x):

    return (np.sqrt((x - tau[0, 0]) ** 2 + (y - tau[1, 0]) ** 2) + a_matrix[0, 0]) * indicator(y, x, (0, 0)) +\
           (np.sqrt((x - tau[0, 1]) ** 2 + (y - tau[1, 1]) ** 2) + a_matrix[1, 0]) * indicator(y, x, (1, 0)) +\
           (np.sqrt((x - tau[0, 2]) ** 2 + (y - tau[1, 2]) ** 2) + a_matrix[2, 0]) * indicator(y, x, (2, 0)) +\
           (np.sqrt((x - tau[0, 3]) ** 2 + (y - tau[1, 3]) ** 2) + a_matrix[3, 0]) * indicator(y, x, (3, 0)) +\
           (np.sqrt((x - tau[0, 4]) ** 2 + (y - tau[1, 4]) ** 2) + a_matrix[4, 0]) * indicator(y, x, (4, 0)) +\
           (np.sqrt((x - tau[0, 5]) ** 2 + (y - tau[1, 5]) ** 2) + a_matrix[5, 0]) * indicator(y, x, (5, 0)) +\
           (np.sqrt((x - tau[0, 6]) ** 2 + (y - tau[1, 6]) ** 2) + a_matrix[6, 0]) * indicator(y, x, (6, 0)) +\
           (np.sqrt((x - tau[0, 7]) ** 2 + (y - tau[1, 7]) ** 2) + a_matrix[7, 0]) * indicator(y, x, (7, 0)) +\
           (np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 0]) * indicator(y, x, (8, 0)) +\
           (np.sqrt((x - tau[0, 0]) ** 2 + (y - tau[1, 0]) ** 2) + a_matrix[0, 1]) * indicator(y, x, (0, 1)) + \
           (np.sqrt((x - tau[0, 1]) ** 2 + (y - tau[1, 1]) ** 2) + a_matrix[1, 1]) * indicator(y, x, (1, 1)) +\
           (np.sqrt((x - tau[0, 2]) ** 2 + (y - tau[1, 2]) ** 2) + a_matrix[2, 1]) * indicator(y, x, (2, 1)) +\
           (np.sqrt((x - tau[0, 3]) ** 2 + (y - tau[1, 3]) ** 2) + a_matrix[3, 1]) * indicator(y, x, (3, 1)) +\
           (np.sqrt((x - tau[0, 4]) ** 2 + (y - tau[1, 4]) ** 2) + a_matrix[4, 1]) * indicator(y, x, (4, 1)) +\
           (np.sqrt((x - tau[0, 5]) ** 2 + (y - tau[1, 5]) ** 2) + a_matrix[5, 1]) * indicator(y, x, (5, 1)) +\
           (np.sqrt((x - tau[0, 6]) ** 2 + (y - tau[1, 6]) ** 2) + a_matrix[6, 1]) * indicator(y, x, (6, 1)) +\
           (np.sqrt((x - tau[0, 7]) ** 2 + (y - tau[1, 7]) ** 2) + a_matrix[7, 1]) * indicator(y, x, (7, 1)) +\
           (np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 1]) * indicator(y, x, (8, 1))


if __name__ == '__main__':

    I = dblquad(dual_target, 0.0, 6.0, lambda x: 0.0, lambda x: 20.0)[0] -\
        np.dot(psi, b_vector)
    print(I)
    I2 = dblquad(target, 0.0, 6.0, lambda x: 0.0, lambda x: 20.0)[0]
    print(I2)
