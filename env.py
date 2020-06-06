import os
import sys
import random
import copy
import json
import numpy as np

"""
train_maze = [
    '..........',
    '...x.x....',
    '..x...x...',
    '......xx..',
    '..xx.xxx..',
    '..x...xx..',
    '..........',
    '..x...xx..',
    '..x...xx..',
    '..........'
]

test_maze = [
    '..........',
    '..........',
    '...xxxx...',
    '....xx....',
    '..........',
    '....xx....',
    'xx..xx..xx',
    '..........',
    '..xx..xx..',
    '..........'
]
"""

train_maze = [
    '.....',
    '.....',
    '.....',
    '.....',
    '.....'
]

test_maze = copy.deepcopy(train_maze)


class Environment(object):

    action_space = [(1, 0), (0, 1)]
    symbol_indices = { '.' : 0., 'x' : -1., 'O': 1., '#': 2., 'G': 3 }

    def __init__(self, mode, teacher, args):

        self.mode = mode
        self._reset_maze()
        self.w = len(self.maze[0])
        self.h = len(self.maze)

        self.test_cases = [None] * 1000

        self.teacher = teacher
        self.args = args

        self.reset()
        self.reset_record()
        self.d = self.shortest_path()

    def reset_record(self):
        self.nav_proba = [[0] * self.w for _ in range(self.h)]
        self.query_count = [[0] * self.w for _ in range(self.h)]
        self.query_proba = [[0] * self.w for _ in range(self.h)]
        self.vis_count = [[0] * self.w for _ in range(self.h)]
        self.intrinsic = [[0] * self.w for _ in range(self.h)]
        self.extrinsic = [[0] * self.w for _ in range(self.h)]


    def input_size(self):
        return len(self.maze) * len(self.maze[0])

    def num_actions(self):
        return len(self.action_space)

    def render(self):
        print()
        for r in self.maze:
            print('    ', ''.join(r))

    def reset(self, test_id=None):

        self._reset_maze()

        self.agent_pos = (0, 0)
        self.maze[self.agent_pos[0]][self.agent_pos[1]] = 'O'

        goal_pos = (self.h - 1, self.w - 1)
        self.goal_pos = goal_pos
        self.maze[goal_pos[0]][goal_pos[1]] = 'G'

        self._calculate_valid_actions()

        return self._get_ob()

    def _reset_maze(self):
        if 'train' in self.mode:
            maze = train_maze
        elif 'test' in self.mode:
            maze = test_maze
        else:
            raise ValueError('Invalid mode! %s' % self.mode)

        self.maze = []
        for r in maze:
            self.maze.append(list(r))

    def _get_ob(self):
        ob = []
        for r in self.maze:
            for c in r:
                ob.append(self.symbol_indices[c])
        return ob

    def shortest_path(self):

        d = {}

        queue = [None] * 1000
        start = 0
        end = 0
        queue[end] = self.goal_pos
        end += 1

        d[self.goal_pos] = 0

        while start < end:
            pos = queue[start]
            start += 1

            for i, j in self.action_space:
                new_pos = (pos[0] - i, pos[1] - j)
                if new_pos[0] < 0 or new_pos[0] >= self.h or new_pos[1] < 0 or new_pos[1] >= self.w:
                    continue
                if self.maze[new_pos[0]][new_pos[1]] != 'x' and new_pos not in d:
                    queue[end] = new_pos
                    end += 1
                    d[new_pos] = d[pos] + 1

        return d

    def _calculate_valid_actions(self):
        self.valid_action_indices = []
        for k, (i, j) in enumerate(self.action_space):
            new_pos = (self.agent_pos[0] + i, self.agent_pos[1] + j)
            if new_pos[0] < 0 or new_pos[0] >= self.h or new_pos[1] < 0 or new_pos[1] >= self.w:
                continue
            if self.maze[new_pos[0]][new_pos[1]] != 'x':
                self.valid_action_indices.append(k)

    def step(self, action_idx):

        if action_idx == -1:
            return self._get_ob(), False

        action = self.action_space[action_idx]
        new_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])

        done = self.maze[new_pos[0]][new_pos[1]] == 'G'

        self.maze[self.agent_pos[0]][self.agent_pos[1]] = '.'
        self.agent_pos = new_pos
        self.maze[self.agent_pos[0]][self.agent_pos[1]] = 'O'

        self._calculate_valid_actions()

        return self._get_ob(), done

    def record(self, query_action, uncertainty, query_prob, nav_prob):
        self.query_count[self.agent_pos[0]][self.agent_pos[1]] += query_action
        self.query_proba[self.agent_pos[0]][self.agent_pos[1]] += query_prob
        self.vis_count[self.agent_pos[0]][self.agent_pos[1]] += 1
        self.nav_proba[self.agent_pos[0]][self.agent_pos[1]] += nav_prob
        self.intrinsic[self.agent_pos[0]][self.agent_pos[1]] += uncertainty['intrinsic']
        self.extrinsic[self.agent_pos[0]][self.agent_pos[1]] += uncertainty['extrinsic']
