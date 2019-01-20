import psycopg2


if __name__ == '__main__':
    conn = psycopg2.connect('dbname=optimization user=postgres password=postgres_ivan')
    cur = conn.cursor()
    cur.execute('select * from sample')
    sample_data = cur.fetchall()
    for part_of_data in sample_data:
        elem_id, elem_name = part_of_data[0], part_of_data[1]
        x = ['' for _ in range(6)]
        for i in range(len(x)):
            if part_of_data[i + 2] >= 8:
                x[i] = 'В'
            elif part_of_data[i + 2] >= 4:
                x[i] = 'С'
            else:
                x[i] = 'Н'
        if part_of_data[8] >= 8:
            y = 'В'
            y_class = 'D1'
            y_class_value = 0
        elif part_of_data[8] >= 4:
            y = 'С'
            y_class = 'D2'
            y_class_value = 5
        else:
            y_class = 'D3'
            y = 'Н'
            y_class_value = 10
        cur.execute("insert into sample_with_classes values (%d, '%s', '%s', '%s', '%s', '%s',"
                    "'%s', '%s', '%s', '%s', %d)" %
                    (elem_id, elem_name, x[0], x[1], x[2], x[3], x[4], x[5], y, y_class, y_class_value))
    conn.commit()
    cur.close()
    conn.close()
