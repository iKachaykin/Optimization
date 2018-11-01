import numpy as np
from NonLinearOptimization_old import grad
from NonLinearOptimization_old import left_side_grad
from NonLinearOptimization_old import right_side_grad
from scipy.optimize import approx_fprime as lib_grad
from numpy.linalg import norm
from scipy.optimize import rosen


def f1(x):
    return norm(x, axis=0)


if __name__ == '__main__':
    for x in np.linspace(-np.pi, np.pi, 10):
        for y in np.linspace(-np.pi, np.pi, 10):
            x0 = np.array([x, y])
            exact_grad = np.array([x / norm([x, y]), y / norm([x, y])])
            calc_grad = right_side_grad(x0, f1)
            print(norm(exact_grad - calc_grad))
