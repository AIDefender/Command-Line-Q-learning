'''
Implementation of the most primary Q-learning in Grid World environment

@author Zheng H. Xue, School of Artificial Intelligence, Nanjing University


The environment describes a cave where large amount of money is surrounded by two devils. The agent must navigate
itself to avoid the devil and reach the money.

Originally, the agent has no prior knowledege of the dynamics of the environment, i.e. what influence will
be exerted if it takes a certain action. The agent learns the environment in a reinforcement learning setting,
taking random actions at first and modifying itself via rewards previously set, which are: small negative number
for each step it takes, large negative number if it touches the devil and large positive number if it touches
the goal.

The replay buffer, which is a standard practice in deep Q-learning, was implemented but proved unnecessary 
and bug-invoking during test, so it was abandoned. Therefore the algorithm is an on-policy Q-learning


Note: No use of deep Q-learning, deep neural networks or deep learning frameworks
'''


import argparse
from QLearningAgent import QLearningAgent
from env import Env
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--num_epoches', '-n', type=int, default=250)
parser.add_argument('--learning_rate', '-lr', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=1314)
parser.add_argument('--render_freq', type=int, default=25)
parser.add_argument('--gamma', '-g', type=float, default=0.9)
parser.add_argument('--epsilon', '-e', type=float, default=1)
parser.add_argument('--epsilon_decay_rate', type=float, default=0.9)
parser.add_argument('--epsilon_decay_freq', type=int, default=50)
parser.add_argument('--least_epsilon', type=float, default=0.05)
parser.add_argument('--layout', type=str, default='easy')
args = parser.parse_args()

env = Env(args.layout)


def train(env):
    agent = QLearningAgent(args.seed,
                           lr=args.learning_rate,
                           gamma=args.gamma,
                           epsilon_param=[args.epsilon,
                                          args.epsilon_decay_rate,
                                          args.epsilon_decay_freq,
                                          args.least_epsilon])
    for i in range(args.num_epoches):
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, done, next_state)
            state = copy.deepcopy(next_state)
            if done:
                break

        if i % args.render_freq == 0:
            agent.apply_policy(env, epoch=i, render=True)


if __name__ == '__main__':
    train(env)
