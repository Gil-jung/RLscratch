import copy
from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ch08.replay_buffer import ReplayBuffer


class QNet(nn.Module):   # 신경망 클래스
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(in_features=4, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=128)
        self.l3 = nn.Linear(in_features=128, out_features=action_size)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DQNAgent:  # 에이전트 클래스
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000  # 경험 재생 버퍼 크기
        self.batch_size = 32      # 미니배치 크기
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)         # 원본 신경망
        self.qnet_target = QNet(self.action_size)  # 목표 신경망
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = state[np.newaxis, :]  # 배치 처리용 차원 추가
            qs = self.qnet(torch.from_numpy(state))
            return qs.argmax()
    
    def update(self, state, action, reward, next_state, done):
        # 경험 재생 버퍼에 경험 데이터 추가
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return   # 데이터가 미니배치 크기만큼 쌓이지 않았다면 여기서 끝
        
        # 미니배치 크기 이상이 쌓이면 미니배치 생성
        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(torch.from_numpy(state))
        q = qs[np.arange(self.batch_size), action]

        next_qs = self.qnet_target(torch.from_numpy(next_state))
        next_q = next_qs.max(axis=1)[0]
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss()
        loss = loss_fn(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def sync_qnet(self):  # 두 신경망 동기화
        self.qnet_target = copy.deepcopy(self.qnet)