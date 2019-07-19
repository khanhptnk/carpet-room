import os
import sys
import random
import argparse
import time

import torch

import flags
from env import *
from teacher import *
from architecture import *
from agent import *


def get_time_str(start_time):
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)


parser = flags.make_parser()
args = parser.parse_args()

n_train    = args.n_train
test_every = args.test_every
n_test     = args.n_test

random.seed(241)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

if args.feedback == 'reward':
    teacher = RewardTeacher()
elif args.feedback == 'action':
    teacher = ActionTeacher()

train_env = Environment('train_' + args.mode, teacher, args)
test_env  = Environment('test_' + args.mode, teacher, args)
model = ReinforceModel(train_env.input_size(), train_env.num_actions(), MLPModel)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.feedback == 'reward':
    agent = ReinforceAgent(model, optimizer)
elif args.feedback == 'action':
    agent = ImitateAgent(model, optimizer)

start_time = time.time()

train_successes = 0
train_agent_reward = 0
train_best_reward = 0
train_loss = 0

for i in range(n_train):

    ep_info = agent.train(train_env)
    train_loss += ep_info['loss']
    train_agent_reward += ep_info['agent_reward']
    train_best_reward += ep_info['best_reward']
    train_successes += ep_info['done']

    if (i + 1) % test_every == 0:
        test_successes = 0
        test_agent_reward = 0
        test_best_reward = 0
        for test_id in range(n_test):
            done, agent_reward, best_reward = agent.test(test_env, test_id=test_id)
            test_successes += done
            test_agent_reward += agent_reward
            test_best_reward += best_reward

        print('%s   Iter %d, Loss %.3f, Train reward: %.2f, Test reward: %.2f, Train reward ratio: %.2f, Test reward ratio: %.2f, Train success: %.2f, Test success: %.2f' %
            (get_time_str(start_time), i + 1, train_loss / test_every,
             train_agent_reward / test_every, test_agent_reward / n_test,
             train_agent_reward / train_best_reward, test_agent_reward / test_best_reward,
             train_successes * 100 / test_every, test_successes * 100 / n_test))

        train_loss = 0
        train_agent_reward = 0
        train_best_reward = 0
        train_successes = 0


