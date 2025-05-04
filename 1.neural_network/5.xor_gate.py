import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


class XorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

x = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

y = torch.tensor([0., 1., 1., 0.]).reshape(-1, 1)

model = XorModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

model.train()
loss_history = []
for epoch in range(5000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()
with torch.no_grad():
    y_pred = model(x)
    loss = criterion(y_pred, y)
    print(f'loss = {loss.item()}')
    for input, output in zip(x, y_pred):
        print(f'{input} => {output}') 


plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(loss_history)
plt.show()

