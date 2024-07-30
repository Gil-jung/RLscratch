import numpy as np
import matplotlib.pyplot as plt
import torch


# 토이 데이터셋
np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x, y = torch.tensor(x), torch.tensor(y)

# 매개변수 정의
W = torch.tensor(np.zeros((1, 1)), requires_grad=True)
b = torch.tensor(np.zeros(1), requires_grad=True)

# 예측 함수
def predict(x):
    y = torch.matmul(x, W) + b  # 행렬의 곱으로 여러 데이터 일괄 계산
    return y

# 평균 제곱 오차(식 7.2) 계산 함수
def mean_squard_error(x0, x1):
    diff = x0 - x1
    return torch.sum(diff ** 2) / len(diff)

# 경사 하강법으로 매개변수 갱신
lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squard_error(y, y_pred)

    loss.backward()
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    W.grad.zero_()
    b.grad.zero_()

    if i % 10 == 0:  # 10회 반복마다 출력
        print(loss.data)

print('====')
print('W =', W.data)
print('b =', b.data)

# [그림 7-9] 학습 후 모델
plt.scatter(x.data, y.data, s=10)
plt.xlabel('x')
plt.ylabel('y')
t = np.arange(0, 1, .01)[:, np.newaxis]
y_pred = predict(torch.tensor(t))
plt.plot(t, y_pred.data, color='r')
plt.show()