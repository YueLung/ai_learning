import torch
import numpy as np

# a = np.array([[[1, 2, 3, 4], [4, 5, 6, 1], [4, 5, 6, 1]],
#               [[1, 2, 3, 1], [4, 5, 6, 1], [4, 5, 6, 1]]])

# a = np.arange(24).reshape(2, 3, 4)
a = np.arange(8).reshape(2,4)

unpermute = torch.tensor(a)
print(unpermute)
print(unpermute.shape)
print(unpermute.stride())

# permute = unpermute.permute(2, 1, 0)
permute = unpermute.permute(1, 0)
print('--- after permute ---')
print(permute)
print(permute.shape)
print(permute.stride())