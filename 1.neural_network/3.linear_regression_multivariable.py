import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html#summary test data

x = torch.tensor(
    [[230.1, 44.5, 17.2, 151.5],
    [37.8, 39.3, 45.9, 41.3],
    [69.1, 23.1, 34.7, 13.2]]).permute(1,0)

# print(x)
# print(x.shape)

y = torch.tensor([22.1, 10.4, 18.3, 18.5]).reshape(-1, 1)

model = nn.Linear(3, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00001)

# print(model(x))

loss_history = []
model.train()
for _ in range(1000):
    y_predicated = model(x)
    loss = criterion(y_predicated, y)
    loss_history.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 推論 + 評估
model.eval()
# 建議用 with torch.no_grad() 來評估  這樣可以避免在測試/推論階段建立計算圖，節省記憶體：
with torch.no_grad():
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print("預測值 vs 真實值")
    for pred, actual in zip(y_pred, y):
        print(f"{pred.item():.2f} vs {actual.item():.2f}")

    print(f'loss = {loss.item():.4f}')    

print(f'w = {model.weight}, b = {model.bias.item()}')

plt.title('loss history')
plt.plot(loss_history)
plt.show()