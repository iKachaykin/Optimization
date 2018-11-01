import psycopg2
import numpy as np


def select_column_from_query(query_result, column):
    func_res = []
    for row in query_result:
        func_res.append(row[column])
    return func_res


if __name__ == '__main__':
    conn = psycopg2.connect('dbname=optimization user=postgres password=postgres_ivan')
    cur = conn.cursor()
    point_class = 2
    cur.execute('select * from optimization.public.numerical_results where point_class = %d' % point_class)
    iters = select_column_from_query(cur.fetchall(), 5)
    iters = np.array(iters)
    print(iters.std())
    conn.commit()
    cur.close()
    conn.close()
