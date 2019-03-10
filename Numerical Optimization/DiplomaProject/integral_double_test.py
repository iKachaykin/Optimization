import numpy as np
import NonLinearOptimization as nlopt
import scipy.integrate as spint
from time import time


if __name__ == '__main__':
    print('SciPy: %.52f\nNLOpt: %.52f' %
          (spint.dblquad(lambda y, x: np.sin(x) + np.cos(y), 0.0, 1.0, 0.0, 20.0)[0],
           nlopt.integral_double(lambda x, y: np.sin(x) + np.cos(y), 0.0, 1.0, 0.0, 20.0, 20, 400)))
