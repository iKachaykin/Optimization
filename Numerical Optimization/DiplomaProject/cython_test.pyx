import numpy as np

cpdef summarize(double[:] a):
    S = 0.0
    for i in range(a.size):
        S += a[i]
    return S


# if __name__ == '__main__':
#     a = np.ones(100000000)
#
#     t1 = time()
#     S1 = sum(a)
#     t1 = time() - t1
#
#     t2 = time()
#     S2 = sum(a)
#     t2 = time() - t2
#
#     print(t1)
#     print(t2)
