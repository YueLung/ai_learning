import torch

# x = torch.tensor(3.0, requires_grad=True)
x = torch.tensor([3.0, 4.0], requires_grad=True)
y = (x**2).mean()
y.backward()

print(f'type = {type(y)}, y = {y}')
print(f"x 的梯度是: {x.grad}")  # 會輸出 tensor(6.)

# =======================================================


# 建立資料
# x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)   # input
x = torch.tensor([1.0, 2.0, 3.0])   # input
y = torch.tensor([5.0, 7.0, 9.0])   # target (真實值)

# 初始化參數：w, b，且要求計算梯度
w = torch.tensor([1.0], requires_grad=True)  # 初始權重
b = torch.tensor([1.0], requires_grad=True)  # 初始偏差

# forward：模型預測 y = wx + b
pred = w * x + b

# MSE Loss：mean squared error
loss = ((pred - y)**2).mean()

# 🔁 backward：從 loss 開始自動反向傳播
loss.backward()

# 印出梯度
print(f"w 的梯度: {w.grad.item():.4f}")
print(f"b 的梯度: {b.grad.item():.4f}")

# print(f"x 的梯度: {x.grad}")
