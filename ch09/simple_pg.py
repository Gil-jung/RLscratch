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


class Policy(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(in_features=4, out_features=128)            # 첫 번째 계층
        self.l2 = nn.Linear(in_features=128, out_features=action_size)  # 두 번째 계층
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class Agent:  # 에이전트 클래스
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0002
        self.action_size = 2

        self.memory = []
        self.pi = Policy(self.action_size)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)
    
    def get_action(self, state):
        state = torch.from_numpy(state[np.newaxis, :])
        probs = self.pi(state)
        probs = probs[0]
        m = Categorical(logits=probs)
        action = m.sample().item()
        return action, m.probs[action]
    
    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)
    
    def update(self):
        G, loss = 0, 0
        for reward, prob in reversed(self.memory):
            G = reward + self.gamma * G
        
        for reward, prob in self.memory:
            loss += - torch.log(prob) * G
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


if __name__ == '__main__':
    episodes = 3000
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

            agent.add(reward, prob)
            state = next_state
            total_reward += reward
        
        agent.update()
        reward_history.append(total_reward)
        if episode % 100 == 0:
            print("episode : {}, total reward : {}".format(episode, total_reward))
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()