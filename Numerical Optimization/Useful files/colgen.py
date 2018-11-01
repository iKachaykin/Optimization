import numpy as np


def add_default_colors(colors, colors_left):
    count, tmp_eye = 0, np.eye(3, 3)
    for i in range(3):
        if colors_left > 0:
            colors.append(tuple(tmp_eye[i]))
            colors_left -= 1
            count += 1
        else:
            return count
    return count


def add_special_colors(colors, colors_left):
    count = 0
    if colors_left > 0:
        colors.append((0.3, 0.0, 0.5))
        colors_left -= 1
        count += 1
    if colors_left > 0:
        colors.insert(1, (1.0, 1.0, 0.0))
        colors_left -= 1
        count += 1
    if colors_left > 0:
        colors.insert(3, (0.0, 1.0, 1.0))
        colors_left -= 1
        count += 1
    return count


def insert_from_r_to_y(colors, number, r_index):
    inserted_colors = np.linspace(0, 1, number, endpoint=False)
    for i in range(number):
        colors.insert(r_index + i + 1, (1, float(inserted_colors[i]), 0))
    return colors


def insert_from_y_to_g(colors, number, y_index):
    inserted_colors = np.linspace(1, 0, number, endpoint=False)
    for i in range(number):
        colors.insert(y_index + i + 1, (float(inserted_colors[i]), 1, 0))
    return colors


def insert_from_g_to_a(colors, number, g_index):
    inserted_colors = np.linspace(0, 1, number, endpoint=False)
    for i in range(number):
        colors.insert(g_index + i + 1, (0, 1, float(inserted_colors[i])))
    return colors


def insert_from_a_to_b(colors, number, a_index):
    inserted_colors = np.linspace(1, 0, number, endpoint=False)
    for i in range(number):
        colors.insert(a_index + i + 1, (0, float(inserted_colors[i]), 1))
    return colors


def insert_from_b_to_i(colors, number, b_index):
    tmp = colors[b_index + 1]
    inserted_colors_1, inserted_colors_2 = \
        np.linspace(0, 0.3, number, endpoint=False), np.linspace(1, 0.5, number, endpoint=False)
    for i in range(number):
        colors.insert(b_index + i + 1, (float(inserted_colors_1[i]), 0, float(inserted_colors_2[i])))
    return colors


def create_colors(number):
    colors_left, colors = number, []
    colors_left -= add_default_colors(colors, colors_left)
    colors_left -= add_special_colors(colors, colors_left)
    if colors_left <= 0:
        return tuple(colors)
    default_colors_indices = np.arange(0, 6)
    insert_colors_functions = (insert_from_r_to_y, insert_from_y_to_g, insert_from_g_to_a,
                               insert_from_a_to_b, insert_from_b_to_i)
    between_default_colors_to_insert_num = [colors_left // 5 for _ in range(5)]
    for i in range(colors_left % 5):
        between_default_colors_to_insert_num[i] += 1
    for i in range(5):
        insert_colors_functions[i](colors, between_default_colors_to_insert_num[i],
                                   int(default_colors_indices[i]))
        default_colors_indices[i + 1:] += between_default_colors_to_insert_num[i]
    return tuple(colors)
