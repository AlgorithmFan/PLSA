'''
Author: Haidong Zhang
E-mail: haidong_zhang13@163.com
'''

#!usr/bin/env python
#coding:utf-8

from random import gammavariate
from random import random
import numpy as np
def normalizeVec(vec):
    s = sum(vec)
    assert(abs(s) != 0.0) # the sum must not be 0

    for i in range(len(vec)):
        assert(vec[i] >= 0) # element must be >= 0
        vec[i] = vec[i] * 1.0 / s

def normalizeMatrix(matrix):
    '''Normalize for each row in this matrix.'''
    row_num, column_num = np.shape(matrix)
    for row_index in range(row_num):
        normalizeVec(matrix[row_index])