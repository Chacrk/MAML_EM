# -*- coding: UTF-8 -*-
from math import sqrt

min_ = 640
min_i = 0
min_j = 0

for i in range(640):
    for j in range(640):
        if i * j == 640:
            if abs(i - j) < min_:
                min_ = abs(i - j)
                min_i = i
                min_j = j

print('i: {}, j: {}'.format(min_i, min_j))