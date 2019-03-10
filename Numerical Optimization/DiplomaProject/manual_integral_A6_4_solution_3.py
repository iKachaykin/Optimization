import numpy as np
from scipy.integrate import dblquad

partition_number, product_number = 9, 3

tau = np.array([
    [3.000001166161274, 3.000005954223865, 3.06602990142659, 1.7066762193848148, 3.000001732016441, 2.9999942831900865,
     3.658490642152464, 3.0000044957735685, 3.0000018455862825],
    [16.06749047761342, 9.650001525798421, 3.912478382504063, 5.082409511342341, 3.2244829172885447, 3.224499067582716,
     12.702469641889437, 9.649999203997053, 5.375432613898625]
])

psi = np.array([-0.00019913742985239803, 8.044618090510326e-05, 4.2933335190631724e-05, 4.2933335190631724e-05,
                9.512794813793294e-05, 0.00017791285754353034, 4.2933335190631724e-05, 0.0001317800917914071,
                3.741068587085819e-09])

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
                    np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 1] + psi[8]]) + \
           np.amin([np.sqrt((x - tau[0, 0]) ** 2 + (y - tau[1, 0]) ** 2) + a_matrix[0, 2] + psi[0],
                    np.sqrt((x - tau[0, 1]) ** 2 + (y - tau[1, 1]) ** 2) + a_matrix[1, 2] + psi[1],
                    np.sqrt((x - tau[0, 2]) ** 2 + (y - tau[1, 2]) ** 2) + a_matrix[2, 2] + psi[2],
                    np.sqrt((x - tau[0, 3]) ** 2 + (y - tau[1, 3]) ** 2) + a_matrix[3, 2] + psi[3],
                    np.sqrt((x - tau[0, 4]) ** 2 + (y - tau[1, 4]) ** 2) + a_matrix[4, 2] + psi[4],
                    np.sqrt((x - tau[0, 5]) ** 2 + (y - tau[1, 5]) ** 2) + a_matrix[5, 2] + psi[5],
                    np.sqrt((x - tau[0, 6]) ** 2 + (y - tau[1, 6]) ** 2) + a_matrix[6, 2] + psi[6],
                    np.sqrt((x - tau[0, 7]) ** 2 + (y - tau[1, 7]) ** 2) + a_matrix[7, 2] + psi[7],
                    np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 2] + psi[8]])


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
           (np.sqrt((x - tau[0, 0]) ** 2 + (y - tau[1, 0]) ** 2) + a_matrix[0, 1]) * indicator(y, x, (0, 1)) +\
           (np.sqrt((x - tau[0, 1]) ** 2 + (y - tau[1, 1]) ** 2) + a_matrix[1, 1]) * indicator(y, x, (1, 1)) +\
           (np.sqrt((x - tau[0, 2]) ** 2 + (y - tau[1, 2]) ** 2) + a_matrix[2, 1]) * indicator(y, x, (2, 1)) +\
           (np.sqrt((x - tau[0, 3]) ** 2 + (y - tau[1, 3]) ** 2) + a_matrix[3, 1]) * indicator(y, x, (3, 1)) +\
           (np.sqrt((x - tau[0, 4]) ** 2 + (y - tau[1, 4]) ** 2) + a_matrix[4, 1]) * indicator(y, x, (4, 1)) +\
           (np.sqrt((x - tau[0, 5]) ** 2 + (y - tau[1, 5]) ** 2) + a_matrix[5, 1]) * indicator(y, x, (5, 1)) +\
           (np.sqrt((x - tau[0, 6]) ** 2 + (y - tau[1, 6]) ** 2) + a_matrix[6, 1]) * indicator(y, x, (6, 1)) +\
           (np.sqrt((x - tau[0, 7]) ** 2 + (y - tau[1, 7]) ** 2) + a_matrix[7, 1]) * indicator(y, x, (7, 1)) +\
           (np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 1]) * indicator(y, x, (8, 1)) +\
           (np.sqrt((x - tau[0, 0]) ** 2 + (y - tau[1, 0]) ** 2) + a_matrix[0, 2]) * indicator(y, x, (0, 2)) +\
           (np.sqrt((x - tau[0, 1]) ** 2 + (y - tau[1, 1]) ** 2) + a_matrix[1, 2]) * indicator(y, x, (1, 2)) +\
           (np.sqrt((x - tau[0, 2]) ** 2 + (y - tau[1, 2]) ** 2) + a_matrix[2, 2]) * indicator(y, x, (2, 2)) +\
           (np.sqrt((x - tau[0, 3]) ** 2 + (y - tau[1, 3]) ** 2) + a_matrix[3, 2]) * indicator(y, x, (3, 2)) +\
           (np.sqrt((x - tau[0, 4]) ** 2 + (y - tau[1, 4]) ** 2) + a_matrix[4, 2]) * indicator(y, x, (4, 2)) +\
           (np.sqrt((x - tau[0, 5]) ** 2 + (y - tau[1, 5]) ** 2) + a_matrix[5, 2]) * indicator(y, x, (5, 2)) +\
           (np.sqrt((x - tau[0, 6]) ** 2 + (y - tau[1, 6]) ** 2) + a_matrix[6, 2]) * indicator(y, x, (6, 2)) +\
           (np.sqrt((x - tau[0, 7]) ** 2 + (y - tau[1, 7]) ** 2) + a_matrix[7, 2]) * indicator(y, x, (7, 2)) +\
           (np.sqrt((x - tau[0, 8]) ** 2 + (y - tau[1, 8]) ** 2) + a_matrix[8, 2]) * indicator(y, x, (8, 2))


if __name__ == '__main__':

    print(a_matrix.T)
    print(b_vector)

    I = dblquad(dual_target, 0.0, 6.0, lambda x: 0.0, lambda x: 20.0)[0] -\
        np.dot(psi, b_vector)
    print(I)
    I2 = dblquad(target, 0.0, 6.0, lambda x: 0.0, lambda x: 20.0)[0]
    print(I2)
