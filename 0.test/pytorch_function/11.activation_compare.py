
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# XOR 資料
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=torch.float32)

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=torch.float32)

# 定義模型
def get_model():
    return nn.Sequential(
        nn.Linear(2, 4),
        nn.Tanh(),
        nn.Linear(4, 1),
        nn.Sigmoid()
    )

# 訓練函式
def train(optimizer_class, label, lr=0.01):
    model = get_model()
    optimizer = optimizer_class(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    losses = []

    for epoch in range(3000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses

# 訓練各種 Optimizer
optimizers = {
    'SGD': lambda: train(torch.optim.SGD, 'SGD', lr=0.1),
    'Adam': lambda: train(torch.optim.Adam, 'Adam', lr=0.01),
    'AdamW': lambda: train(torch.optim.AdamW, 'AdamW', lr=0.01),
    'RMSprop': lambda: train(torch.optim.RMSprop, 'RMSprop', lr=0.01),
}

# 畫圖
plt.figure(figsize=(10, 6))
for name, trainer in optimizers.items():
    losses = trainer()
    plt.plot(losses, label=name)

plt.title('Optimizer Comparison on XOR')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
