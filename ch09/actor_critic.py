if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.utils import plot_total_reward


class PolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(in_features=4, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=action_size)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(in_features=4, out_features=128)
        self.l2 = nn.Linear(in_features=128, out_features=1)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet(self.action_size)
        self.v = ValueNet()

        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)
    
    def get_action(self, state):
        state = torch.from_numpy(state[np.newaxis, :])
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(logits=probs)
        action = m.sample().item()
        return action, m.probs[action]
    
    def update(self, state, action_prob, reward, next_state, done):
        state = torch.from_numpy(state[np.newaxis, :])
        next_state = torch.from_numpy(next_state[np.newaxis, :])

        target = reward + self.gamma * self.v(next_state) * (1 - done)
        target.detach()
        v = self.v(state)
        loss_fn = nn.MSELoss()
        loss_v = loss_fn(v, target)

        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()


if __name__ == '__main__':
    episodes = 2000
    env = gym.make('CartPole-v0', render_mode='rgb_array')
    agent = Agent()
    reward_history = []

    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0

        while not done:
            action, prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated

            agent.update(state, prob, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode : {}, total reward : {}".format(episode, total_reward))
    
    plot_total_reward(reward_history)