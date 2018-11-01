import numpy as np
from numpy.random import rand
from numpy.linalg import det
from numpy.linalg import inv
from tqdm import tqdm


if __name__ == '__main__':
    # dimension_of_A, experiment_number, a, b = 1000, 100000, -10, 10
    # count = 0
    # for i in tqdm(range(experiment_number), ncols=100):
    #     A = rand(dimension_of_A, dimension_of_A) * (b - a) + a
    #     if np.abs(det(A + A.T)) < 1e-15:
    #         count += 1
    # print('A number is %d: ' % count)

    dimension_of_A, experiment_number, a, b = 1000, 10, -10, 10
    count = 0
    for i in tqdm(range(experiment_number), ncols=100):
        A = rand(dimension_of_A, dimension_of_A) * (b - a) + a
        for j in range(dimension_of_A - 1):
            if det(np.dot(np.dot(A, np.diag(np.arange(1, dimension_of_A + 1))), inv(A))[:j+1, :j+1]) < 1e-8:
                count += 1
                break
    print('A number is %d: ' % count)
