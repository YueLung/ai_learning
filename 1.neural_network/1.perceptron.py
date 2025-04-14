import numpy as np


def step_function(x):
    return 1 if x >= 0 else 0


def perceptron(x, w, b):
    total = np.dot(x, w) + b
    return step_function(total)


# 測試
x = np.array([1, 0])  # 輸入
w = np.array([0.5, 0.5])  # 權重
b = -0.4  # 偏差值

print(perceptron(x, w, b))  # 結果應該是 1

"""
   x1 ----\
           \
   x2 -----> (∑ weighted sum) --> activation --> y
           /
   xn ----/

"""
