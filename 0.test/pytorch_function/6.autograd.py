import torch

# 定義一個 tensor，並開啟 autograd
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3

# 計算 z 的相對於 x 的梯度
grads = torch.autograd.grad(z.sum(), x)
print(grads)


'''
torch.autograd.grad() 是用來計算某些指定輸出相對於輸入的梯度（partial derivatives）。

它返回的是一個元組（tuple），包含了指定輸出的梯度。

可以讓你更精確地控制計算梯度的過程，並且它不會直接修改模型參數的 .grad 屬性，而是返回計算的結果。

用途：當你需要計算某些特定層或變數的梯度，而不會自動更新 .grad 屬性時，這是一個好選擇。
'''