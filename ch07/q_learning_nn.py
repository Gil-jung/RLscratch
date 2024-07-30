if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]

state = (2, 0)
x = one_hot(state)

print(x.shape)
print(x)

# # 데이터셋 생성
# np.random.seed(0)
# x = np.random.rand(100, 1)
# y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
# x, y = torch.tensor(x), torch.tensor(y)

# lr = 0.2
# iters = 10000

# class TwoLayerNet(nn.Module):
#     def __init__(self, in_size, hidden_size, out_size):
#         super().__init__()
#         self.l1 = nn.Linear(in_features=in_size, out_features=hidden_size, dtype=float)
#         self.l2 = nn.Linear(in_features=hidden_size, out_features=out_size, dtype=float)
    
#     def forward(self, x):
#         y = F.sigmoid(self.l1(x))
#         y = self.l2(y)
#         return y

# model = TwoLayerNet(1, 10, 1)
# optimizer = optim.SGD(params=model.parameters(), lr=lr)
# loss_fn = nn.MSELoss()

# for i in range(iters):
#     y_pred = model(torch.tensor(x))
#     loss = loss_fn(y, y_pred)

#     optimizer.zero_grad()
#     loss.backward()

#     optimizer.step()
#     if i % 1000 == 0:  # 1000회 반복마다 출력
#         print(loss.data)

# # 그래프로 시각화([그림 7-12]와 같음)
# plt.scatter(x.data, y.data, s=10)
# plt.xlabel('x')
# plt.ylabel('y')
# t = np.arange(0, 1, .01)[:, np.newaxis]
# y_pred = model(torch.tensor(t))
# plt.plot(t, y_pred.data, color='r')
# plt.show()