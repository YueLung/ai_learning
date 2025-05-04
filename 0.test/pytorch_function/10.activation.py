# https://tree.rocks/deep-learning-introduce-pytorch-activation-function-7c8da0a78dc6

import torch
import matplotlib.pyplot as plt


# x = torch.arange(-10., 10., step=0.01)
x = torch.arange(-10000., 10000., step=0.01)

plt.subplot(3, 1, 1)
plt.title('sigmoid')
plt.plot(x, torch.sigmoid(x))

plt.subplot(3, 1, 2)
plt.title('relu')
plt.plot(x, torch.relu(x))

plt.subplot(3, 1, 3)
plt.title('tanh')
plt.plot(x, torch.tanh(x))

plt.tight_layout()
plt.show()
