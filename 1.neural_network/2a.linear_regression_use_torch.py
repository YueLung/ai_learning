import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

x = torch.tensor([3.75, 9.51, 7.32, 5.99, 1.56, 1.56, 0.58, 8.66, 6.01, 7.08,
     2.12, 9.70, 8.33, 2.90, 1.46, 6.12, 3.92, 1.83, 3.04, 5.25,
     4.32, 0.87, 3.26, 9.01, 2.66, 1.08, 3.36, 3.67, 6.92, 0.87]).reshape(-1, 1)

y = torch.tensor([10.08, 24.68, 19.20, 15.61, 5.15, 3.98, 3.11, 22.32, 16.69, 18.89,
     6.41, 25.08, 22.73, 8.48, 3.84, 17.34, 11.44, 6.45, 8.37, 14.22,
     11.60, 2.46, 9.73, 24.24, 8.42, 3.93, 10.44, 9.49, 19.94, 3.12]).reshape(-1, 1)

# x = torch.tensor([1.,3.,5.]).reshape(-1, 1)
# y = torch.tensor([2.,6.,10.]).reshape(-1, 1)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# pred = model(x)
# print(type(pred))
# loss = criterion(pred, y)
# print(loss.item())
# print(type(loss.item()))
# print(type(loss))
# for p in model.parameters():
#     print(p.shape)


loss_history = []

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, y)
    loss_history.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("w:", model.weight.item(), "b:", model.bias.item(), 'loss: ', criterion(pred, y).item())

x_line = np.linspace(min(x), max(x), 100)
y_line = model.weight.item() * x_line + model.bias.item()

# plt.figure(figsize=(8, 6))

plt.subplot(2, 1, 1)
plt.title('')
plt.plot(x, y, 'o')
plt.plot(x_line, y_line)

plt.subplot(2, 1, 2)
plt.title('loss history')
plt.plot(loss_history)
plt.tight_layout()
plt.show()

pred = model(x).detach()
loss = criterion(pred, y)
print(loss.item())