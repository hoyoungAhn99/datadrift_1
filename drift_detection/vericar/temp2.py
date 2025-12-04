import torch

a = torch.stack([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])

m = torch.min(a, 3.0)

print(m)