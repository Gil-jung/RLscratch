import numpy as np
import torch


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y

x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

iters = 10000  # 반복 횟수
lr = 0.001     # 학습률

for i in range(iters):  # 갱신 반복
    y = rosenbrock(x0, x1)

    # 미분(역전파)
    y.backward()

    # 변수 갱신
    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

    # 이전 반복에서 더해진 미분 초기화
    x0.grad.zero_()
    x1.grad.zero_()

print(x0, x1)