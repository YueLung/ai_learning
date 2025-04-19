import torch

a = torch.arange(6)       # tensor([0, 1, 2, 3, 4, 5])
b = a.reshape(2, 3)       # 2 行 3 列
print(b)


# 自動推算尺寸
a = torch.arange(12)
b = a.reshape(-1, 4)      # PyTorch 自動推算「幾行」
print(b)

a = torch.arange(9).view(3, 3)
b = a.permute(1, 0)         # 打亂記憶體順序
print(b.storage().data_ptr())
c = b.reshape(9, 1)         # reshape 正常  深copy
print(c.storage().data_ptr())

b[0] = 100
print(c)

# d = b.view(9, 1)            # 如果用 view 這裡會報錯
c[0]= 100
# print(d)

# 

'''
🔧 想快速改 tensor 形狀 ➜ 用 reshape()

🧼 確定是連續的 memory ➜ view() 速度更快一點點

😵 有遇到 .view() 報錯 ➜ 換 reshape() 就對了
'''

# https://blog.csdn.net/Flag_ing/article/details/109129752