import os
import sys
import random
import argparse
import time
from pprint import pprint

import torch

import flags
from env import *
from teacher import *
from architecture import *
from active_agent import *


def get_time_str(start_time):
    end_time = time.time()
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)


parser = flags.make_parser()
args = parser.parse_args()

print()
print(args)
print()

n_train    = args.n_train
test_every = args.test_every

random.seed(241)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

teacher = ActionTeacher(args.teacher_type)

train_env = Environment('train_' + args.mode, teacher, args)
test_env  = Environment('test_' + args.mode, teacher, args)

model = ActiveImitateModel(train_env.input_size(), train_env.num_actions(), args)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
agent = ActiveImitateAgent(model, teacher, optimizer, args)

start_time = time.time()

train_successes = 0
train_loss = 0
num_queries = 0
total_steps = 0
uncertainty = { 'intrinsic' : 0, 'extrinsic' : 0, 'model': 0 }

for i in range(n_train):

    ep_info         = agent.train(train_env)
    train_loss      += ep_info['loss']
    train_successes += ep_info['done']
    num_queries        += ep_info['num_queries']
    total_steps     += ep_info['total_steps']
    uncertainty['intrinsic'] += ep_info['intrinsic']
    uncertainty['extrinsic'] += ep_info['extrinsic']
    uncertainty['model'] += ep_info['model']

    if (i + 1) % test_every == 0:
        print('%s   Iter %d, Loss %.3f, Train success: %.2f, Ask ratio: %.2f' %
                (get_time_str(start_time), i + 1, train_loss / test_every,
                 train_successes * 100 / test_every, num_queries / total_steps))
        print('              Total: %.5f, Intrinsic: %.5f, Extrinsic: %.5f, Model: %.5f' %
                (
                    (uncertainty['intrinsic'] + uncertainty['extrinsic']) / total_steps,
                    uncertainty['intrinsic'] / total_steps,
                    uncertainty['extrinsic'] / total_steps,
                    uncertainty['model'] / total_steps
                )
            )


        #num_queries = 0
        #total_steps = 0
        #train_loss = 0
        #train_successes = 0

        print()
        for i in range(len(train_env.vis_count)):
            for j in range(len(train_env.vis_count[i])):
                value = train_env.vis_count[i][j]
                print('%6s' % value, end='')
            print()
        print()

        for i in range(len(train_env.query_count)):
            for j in range(len(train_env.query_count[i])):
                print('%6.2f' % (train_env.query_count[i][j] /
                                (train_env.vis_count[i][j] + 1e-5)), end='')
            print()
        print()

        for i in range(len(train_env.query_proba)):
            for j in range(len(train_env.query_proba[i])):
                value = train_env.query_proba[i][j] / (train_env.vis_count[i][j] + 1e-9)
                print('%10.5f' % value, end='')
            print()
        print()

        for i in range(len(train_env.nav_proba)):
            for j in range(len(train_env.nav_proba[i])):
                value = train_env.nav_proba[i][j] * 100 / (train_env.vis_count[i][j] + 1e-9)
                print('%10.1f |%5.1f' % (value, 100 - value), end='')
            print()
        print()

        for i in range(len(train_env.nav_proba)):
            for j in range(len(train_env.nav_proba[i])):
                value = train_env.intrinsic[i][j] / (train_env.vis_count[i][j] + 1e-9)
                print('%10.5f' % value, end='')
            print()
        print()

        for i in range(len(train_env.nav_proba)):
            for j in range(len(train_env.nav_proba[i])):
                value = train_env.extrinsic[i][j] / (train_env.vis_count[i][j] + 1e-9)
                print('%10.5f' % value, end='')
            print()
        print()


        #train_env.reset_record()


        time.sleep(0.5)


