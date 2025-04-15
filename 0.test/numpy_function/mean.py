import numpy as np

a = [1, 2, 3, 4, 5]
mean_val = np.mean(a)
print(mean_val)  # 輸出: 3.0


arr = np.array([[1, 2, 3],
                [4, 5, 6]])
# 整體平均
print(np.mean(arr))  # 輸出: 3.5

# 每欄平均（axis=0）
print(np.mean(arr, axis=0))  # 輸出: [2.5 3.5 4.5]

# 每列平均（axis=1）
print(np.mean(arr, axis=1))  # 輸出: [2. 5.]