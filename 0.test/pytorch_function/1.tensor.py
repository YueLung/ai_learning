import torch

# 建立 Tensor
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([3.0, 2.0, 1.0])

# 基本運算
print(a + b)        # 加法
print(a * b)        # 乘法
print(torch.dot(a, b))  # 點積
print(a.mean())     # 平均
