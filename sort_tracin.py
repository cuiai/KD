import numpy as np
import operator
# dict = {1: c, 2: d, 3: e}
# np.save('file.npy', dict)
# print(type(dict))
# def sort(new_dict):
#     b = []
#     c = []
#     for i in range(20):
#         a = sorted(new_dict[i].items(), key=operator.itemgetter(1), reverse=True)
#         for k, j in enumerate(a):
#             if k < 10:
#                 print(j)
#             if i == 7000:
#                 break
#             c.append(j[0])
#         b.append(c)
#     return b
#
#
# teacher = {}
def sort(new_list):
    b = []
    d = []
    for i in range(20):
        c = []
        a = sorted(new_list[i], key=lambda x: x[1], reverse=False)
        for k, j in enumerate(a):
            if k < 10:
                print(j)
            if k == 4800:
                break
            c.append(j[0])
        d.append(a)
        b.append(c)
    return d, b
teacher_dict = np.load('error_data/teacher-tracin-value.npy', allow_pickle='TRUE')
# print(len(teacher_dict))
# teacher = sort(teacher_dict)
# student = sort(student_dict)
# inter = set(teacher).intersection(set(student))
# print(len(inter))
m = teacher_dict.tolist()
teacher_sort, teacher_data = sort(m)
teacher_data_new = np.array(teacher_data)
teacher_sort_data = np.array(teacher_sort)
np.save('error_data/teacher_error_data_sort_number_reverse.npy', teacher_data_new)
np.save('error_data/teacher_error_data_sort_reverse.npy', teacher_sort_data)


