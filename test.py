import torch 
x = torch.rand(3, 1)

y = torch.rand(509, 1)

z = torch.cat((x, y), 0)
print(z.size())
print(z)