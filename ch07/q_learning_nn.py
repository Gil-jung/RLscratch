if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.gridworld import GridWorld


def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = np.zeros(HEIGHT * WIDTH, dtype=np.float32)
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec[np.newaxis, :]


class Qnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(in_features=12, out_features=100)  # 중간층의 크기
        self.l2 = nn.Linear(in_features=100, out_features=4)   # 행동의 크기(가능한 행동의 개수)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = Qnet()
        self.optimizer = optim.SGD(self.qnet.parameters(), lr=self.lr)
    
    def get_action(self, state_vec):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = self.qnet(torch.from_numpy(state_vec))
            return qs.argmax()
    
    def update(self, state, action, reward, next_state, done):
        done = int(done)
        next_qs = self.qnet(torch.from_numpy(next_state))
        next_q = next_qs.max(axis=1)[0]
        
        # 목표
        target = reward + ((1 - done) * self.gamma * next_q)
        # 현재 상태에서의 Q 함수 값(q) 계산
        qs = self.qnet(torch.from_numpy(state))
        q = qs[:, action]
        # 목표(target)와 q의 오차 계산
        loss_fn = nn.MSELoss()
        loss = loss_fn(target, q)

        # 역전파 → 매개변수 갱신
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data


env = GridWorld()
agent = QLearningAgent()

episodes = 1000  # 에피소드 수
loss_history = []

for episode in range(episodes):
    print(episode)
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state
    
    average_loss = total_loss / cnt
    loss_history.append(average_loss)


# [그림 7-14] 에피소드별 손실 추이
plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(range(len(loss_history)), loss_history)
plt.show()

# [그림 7-15] 신경망을 이용한 Q 러닝으로 얻은 Q 함수와 정책
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(torch.from_numpy(one_hot(state)))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)