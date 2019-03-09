import numpy as np


def dual_target(y):
    return np.amin([np.sqrt((x-3)**2 + (y-4.221)**2) + 0.0009, np.sqrt((x-4.751)**2 + (y-0.6176)**2) + 100,
                    np.sqrt((x-0.811)**2 + (y-12)**2) + 100, np.sqrt((x-2.0048)**2 + (y-3.3155)**2) + 100,
                    np.sqrt((x-3)**2 + (y-10.525)**2) + 100.0007, np.sqrt((x-0.2081)**2 + (y-17.1092)**2) + 100,
                    np.sqrt((x-4.3896)**2 + (y-10.6873)**2) + 100, np.sqrt((x-3)**2 + (y-16.8376)**2) + 100.0023,
                    np.sqrt((x-3)**2 + (y-14.7376)**2) + 0.0015]) +\
           np.amin([np.sqrt((x-3)**2 + (y-4.221)**2) + 0.0009, np.sqrt((x-4.751)**2 + (y-0.6176)**2) + 100,
                    np.sqrt((x-0.811)**2 + (y-12)**2) + 100, np.sqrt((x-2.0048)**2 + (y-3.3155)**2) + 100,
                    np.sqrt((x-3)**2 + (y-10.525)**2) + 0.0007, np.sqrt((x-0.2081)**2 + (y-17.1092)**2) + 100,
                    np.sqrt((x-4.3896)**2 + (y-10.6873)**2) + 100, np.sqrt((x-3)**2 + (y-16.8376)**2) + 0.0023,
                    np.sqrt((x-3)**2 + (y-14.7376)**2) + 100.0015])
