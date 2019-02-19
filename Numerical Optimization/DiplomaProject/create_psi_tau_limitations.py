if __name__ == '__main__':

    partition_num = 9

    buf = ''

    for i in range(partition_num):
        buf += 'lambda psi, tau: -psi[%i],' % i
        if i % 3 == 2:
            buf += '\n'
        else:
            buf += ' '

    buf += '\n'

    for i in range(partition_num):
        buf += 'lambda psi, tau: x_left - tau[%i],' % i
        if i % 3 == 2:
            buf += '\n'
        else:
            buf += ' '

    buf += '\n'

    for i in range(partition_num):
        buf += 'lambda psi, tau: y_left - tau[%i],' % (partition_num + i)
        if i % 3 == 2:
            buf += '\n'
        else:
            buf += ' '

    buf += '\n'

    for i in range(partition_num):
        buf += 'lambda psi, tau: tau[%i] - x_right,' % i
        if i % 3 == 2:
            buf += '\n'
        else:
            buf += ' '

    buf += '\n'

    for i in range(partition_num):
        buf += 'lambda psi, tau: tau[%i] - y_right' % (partition_num + i)
        if i != partition_num - 1:
            buf += ','
        if i % 3 == 2:
            buf += '\n'
        else:
            buf += ' '

    print(buf)
