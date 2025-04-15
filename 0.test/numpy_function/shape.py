import numpy as np

# 一維陣列
a = np.array([1, 2, 3, 4])
# print("陣列：", a)
print("shape：", a.shape)

# 二維陣列（矩陣）
b = np.array([[1, 2, 3], [4, 5, 6]])
# print("陣列：\n", b)
print("shape：", b.shape)

# 三維陣列
c = np.array([[[1, 2], [3, 4], [3, 4]], [[5, 6], [7, 8], [7, 8]]])
print("shape：", c.shape)

# 用 .reshape() 改變形狀
d = np.array([1, 2, 3, 4, 5, 6])
d2 = d.reshape((2, 3))
print("原始：", d.shape)
print("reshape 後：", d2.shape)
print(d2)
