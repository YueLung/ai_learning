import torch
import torch.nn as nn


model = nn.Linear(2, 1)

input = torch.tensor([1., 2.])
output = model(input)

print(f'input = {input}')
print(f'output = {output}')

# .parameters() is iterator
for param in model.parameters():
    print(param)

# print(model.weight)
# print(model.bias.item())

# -------------------------------

x = torch.tensor([[0.1,0.2,0.3,0.4,0.4],
                  [0.2,0.3,0.4,0.4,0.4],
                  [0.3,0.4,0.3,0.4,0.4]])

model = nn.Linear(in_features= 5, out_features= 10, bias= True)

for param in model.parameters():
    print(param)
print(model.weight.shape) 

# print(model(x).size())   
print(model(x).shape)
# .shape is an alias for .size(),