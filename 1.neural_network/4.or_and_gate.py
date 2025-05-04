import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class AndOrModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sigmoid(),
        )
        # self.linear = nn.Linear(2, 1)
        # self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.model(x)
        # return self.activation(self.linear(x))
    
x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

y = torch.tensor([[0., 1., 1., 1.],
                  [0., 0., 0., 1.]]).permute(1, 0)

# y = torch.tensor([
#     [0., 0.],  # (OR, AND)
#     [1., 0.],
#     [1., 0.],
#     [1., 1.]
# ])

model = AndOrModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
loss_history = []
for epoch in range(3000):
    y_predicted = model(x)
    loss = criterion(y_predicted, y) # 這裡 loss 是一個 scalar，裡面記錄了整個運算圖（computational graph）。
    loss_history.append(loss.item())

    optimizer.zero_grad() # 會清掉 optimizer 裡管理的參數
    # model.zero_grad()   # 會直接清掉 model 內所有有 requires_grad=True 的參數
    
    loss.backward() # 這會沿著計算圖，自動把 loss 對 model 中的參數（例如 w, b）的偏導數算出來，存到 .grad 中。
    optimizer.step() # w = w - lr * ∂loss/∂w
                     # b = b - lr * ∂loss/∂b

model.eval()
with torch.no_grad():
    y_predicted = model(x)
    print(f'loss = {criterion(y_predicted, y)}')
    for input, pred in zip(x, y_predicted):
        or_pred, and_pred = pred
        print(f'{input.tolist()} => or => {or_pred.item():.4f}, and => {and_pred.item():.4f}')

for p in model.parameters():
    print(p)

plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')

# plt.show()