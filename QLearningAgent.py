import tensorflow as tf
import numpy as np
from env import Env
import time
import os
import copy


class QLearningAgent(object):
    
    """The agent implementing Q-learning"""

    def __init__(self, seed, max_step=200, size=4, lr=0.1, gamma=0.99, epsilon_param=[0.7, 0.9, 50, 0.05]):
        self.count = 0
        self.size = 4
        self.seed = seed
        self.max_step = max_step
        np.random.seed(seed=seed)
        self.lr = lr
        self.gamma = gamma
        self.tabular_Q = np.zeros((size, size, 4))
        self.epsilon = epsilon_param[0]
        self.epsilon_decay_rate = epsilon_param[1]
        self.epsilon_decay_freq = epsilon_param[2]
        self.least_epsilon = epsilon_param[3]

    def choose_action(self, state):
        self.count += 1
        if self.count % self.epsilon_decay_freq == 0 and self.epsilon > self.least_epsilon:
            self.epsilon *= self.epsilon_decay_rate
        # epsilon-greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.size)
        else:
            a = np.max(self.tabular_Q[int(state[0]), int(state[1]), :])
            b = []
            for i in range(4):
                if self.tabular_Q[int(state[0]), int(state[1]), int(i)] == a:
                    b.append(i)
            return int(np.random.choice(b))

    def update(self, state, action, reward, done, next_state):
        if not done:
            self.tabular_Q[int(state[0]), int(state[1]), int(action)] +=\
                self.lr*(reward*self.gamma+np.max(self.tabular_Q[int(next_state[0]), int(next_state[1]), :]) -
                         self.tabular_Q[int(state[0]), int(state[1]), int(action)])
        else:
            self.tabular_Q[int(state[0]), int(state[1]),
                           int(action)] = reward*self.gamma
    def apply_policy(self, env, epoch, render=False):
        os.system('clear')
        state = env.reset()
        total_reward = 0
        if render:
            print('epoch{}'.format(epoch))
            env.render()
        time.sleep(1)
        os.system('clear')
        step = 0
        while True:
            action = self.choose_action(state)
            next_state, reward, done = env.step(action)
            step += 1
            total_reward += reward
            if done:
                if render and reward > 0:
                    print('Mission Completed!!!Total step:{}'.format(step))
                    time.sleep(1)
                    os.system('clear')
                if render and reward < 0:
                    print('Devil touched. You are dead!!!')
                    time.sleep(1)
                    os.system('clear')
                return total_reward
            state = copy.deepcopy(next_state)
            if render:
                print('epoch{}'.format(epoch))
                env.render()
                self.render_tabular_Q()
                time.sleep(1)
                os.system('clear')

    def render_tabular_Q(self):
        print('epsilon:{}'.format(self.epsilon))
        print('The Q value of each action at each step:')
        for i in range(self.size):
            for j in range(self.size):
                print('Up:%.1f, Down:%.1f, Left:%.1f, Right:%.1f' % (self.tabular_Q[i, j, 0], self.tabular_Q[i, j, 1],
                                                       self.tabular_Q[i, j, 2], self.tabular_Q[i, j, 3]),
                      end='  ')
            print(' ')
        print(' ')
