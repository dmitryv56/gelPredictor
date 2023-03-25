#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import random
import gc

def testInv():
    order = 144
    n_size = 160
    X = np.array([[random.random() for i in range(order)] for j in range(n_size)])
    a = np.zeros((order, order), dtype=float)
    for k in range(order):
        for l in range(order):
            a[k][l]=0.0
            for i in range(n_size):
                a[k][l]=a[k][l] + X[i][k]*X[i][l]
            a[k][l]=a[k][l]/float(order)
    pass
    del(X)
    gc.collect()
    # for i in range(order):
    #     for j in range(order):
    #         a[i][j] = 1.0 if i == j else  float(abs((order -i)-j))/float(order)
    # print("{} {} {}".format(a[0][0], a[0][1], a[0][2]))
    # print("{} {} {}".format(a[1][0], a[1][1], a[1][2]))
    # print(a)
    b=np.linalg.inv(a)
    # print("{} {} {}".format(b[0][0],b[0][1],b[0][2]))
    # print(b)
    c = np.matmul(a,b)
    print(c[0][0], c[0][1],c[0][order-1])
    print(c[1][0], c[1][1], c[1][order - 1])
    print(c[order-1][0], c[order-1][1], c[order - 1][order - 1])
if __name__ == "__main__":
    testInv()

    pass