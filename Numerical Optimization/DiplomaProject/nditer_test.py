import numpy as np
from cython_test import summarize
from time import time


if __name__ == '__main__':

    a = np.zeros(1000000000)

    t1 = time()
    S1 = 0.0
    for i in range(a.size):
        S1 += a[i]
    t1 = time() - t1
    print('Python loop: %f' % t1)

    t2 = time()
    S2 = summarize(a)
    t2 = time() - t2
    print('Cython loop: %f' % t2)

    t3 = time()
    S3 = a.sum()
    t3 = time() - t3
    print('NumPy: %f' % t3)
