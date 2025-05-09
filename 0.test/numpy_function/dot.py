import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)
print("向量點積結果：", result)
# 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
# --------------------------------------------

A = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([7, 8])

result = np.dot(A, x)
print("矩陣乘以向量：", result)
"""
[1*7 + 2*8] = 23
[3*7 + 4*8] = 53
[5*7 + 6*8] = 83
"""
# --------------------------------------------

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

result = np.dot(A, B)
print("矩陣乘法結果：\n", result)
# --------------------------------------------

A = np.random.rand(10, 3)  # 10 個樣本，每個 3 特徵
w = np.array([0.2, 0.5, 0.3])  # 權重

y = np.dot(A, w)
print("加權總和 y.shape =", y.shape)
# --------------------------------------------
