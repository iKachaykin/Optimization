from cx_Freeze import setup, Executable


if __name__ == '__main__':
    executables = [Executable('/Users/ivankachaikin/PycharmProjects/NumericalOptimization/r_algorithm.py')]

    setup(name='r_algorithm', version='0.0.1', executables=executables)
