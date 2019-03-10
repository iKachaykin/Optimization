import numpy as np
import numpy.polynomial.legendre as leg
import tqdm


if __name__ == '__main__':
    roots = []
    for i in tqdm.tqdm(range(2, 1002)):
        legc = np.zeros(i)
        legc[-1] = 1.0
        roots.append(leg.legroots(legc))

    buf = '_leg_roots = [\n\n'
    for r in roots:
        buf += 'np.array(['
        for i in range(len(r)):
            buf += '%.52f' % r[i]
            if i == len(r) - 1:
                buf += '])'
            else:
                buf += ', '
        buf += ',\n\n'

    f = open('legendre_roots.txt', 'w')
    f.write(buf)
    f.close()
