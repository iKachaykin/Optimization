import numpy as np
import NonLinearOptimization_cython as nlopt
import scipy.integrate as spint
from time import time


if __name__ == '__main__':
    print('SciPy: %f\nNLOpt: %f' %
          (spint.dblquad(lambda y, x: np.exp(-x**2 - y**2), 0.0, 1.0, 0.0, 20.0)[0],
           nlopt.trapezoid_double(lambda x, y: np.exp(-x**2 - y**2), 0.0, 1.0, 0.0, 20.0, 1000, 20000)))

    t1 = time()

