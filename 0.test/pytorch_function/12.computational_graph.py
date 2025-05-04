# https://www.cnblogs.com/yanqiang/p/12771928.html
import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w, x)     # retain_grad()
b = torch.add(w, 1)
y = torch.mul(a, b)

y.backward()
print(w.grad) # tensor([5.])

# 查看叶子结点
print("is_leaf:\n", w.is_leaf, x.is_leaf, a.is_leaf, b.is_leaf, y.is_leaf)
# is_leaf: True True False False False

# 查看梯度
print("gradient:\n", w.grad, x.grad, a.grad, b.grad, y.grad)
# gradient: tensor([5.]) tensor([2.]) None None None

# 查看 grad_fn
print("grad_fn:\n", w.grad_fn, x.grad_fn, a.grad_fn, b.grad_fn, y.grad_fn)
# grad_fn:
# None 
# None 
# <AddBackward0 object at 0x00000258F55C28D0> 
# <AddBackward0 object at 0x00000258F55C2A58> 
# <MulBackward0 object at 0x00000258F55D5518>