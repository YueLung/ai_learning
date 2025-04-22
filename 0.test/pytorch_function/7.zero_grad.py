import torch

x = torch.tensor([2.0], requires_grad=True)

# 第一次計算
y = x ** 2
y.backward()
print(x.grad)  # 4.0

# 第二次又計算一次（沒清梯度！）
z = x ** 3
z.backward()
print(x.grad)  # 4.0 + 12.0 = 16.0 ❌ 梯度累加了！

# 清除梯度
x.grad.zero_()
zz = x ** 3
zz.backward()
print(x.grad)
