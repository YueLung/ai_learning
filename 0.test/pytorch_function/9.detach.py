import torch

a = torch.tensor([1.0, 2.0, 3.0], requires_grad = True)
# print(a)

b = a.detach() # set requires_grad = false
b[0] = 100.0

print(a)
print(b)

# 實際上，detach()就是返回一个新的tensor，並且這個tensor是從當前的計算圖中分離出來的。
# 但是返回的tensor和原來的tensor是共享內存的。

# https://zhuanlan.zhihu.com/p/410199046