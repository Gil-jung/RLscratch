import numpy as np
import torch
import torch.nn.functional as F

# 벡터의 내적
a, b = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])
c = torch.matmul(a, b)
print(c)

# 행렬의 곱
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
c = torch.matmul(a, b)
print(c)