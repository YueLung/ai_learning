import torch

a = torch.arange(6)  # tensor([0, 1, 2, 3, 4, 5])
b = a.view(2, 3)     # 2行3列
a[0]=100 # a,b 共享storage
print(b)

# 自動推算尺寸（用 -1）
a = torch.arange(12)
b = a.view(-1, 4)  # 自動推算行數，4列
print(b)

# 2D → 1D
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = a.view(-1)  # 拉平成一維
print(b)  # tensor([1, 2, 3, 4, 5, 6])


'''
.view() 需要原 tensor 是 contiguous（連續記憶體），否則會報錯。

若出錯可以改用 .reshape()，比較彈性。
'''
