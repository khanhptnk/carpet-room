import os
import sys
import random
import copy
import json
import numpy as np


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


class Environment(object):

    action_space = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    symbol_indices = { '.' : 0., 'x' : -1., 'O': 1., '#': 2., 'G': 3 }

    def __init__(self, mode, teacher, args):

        self.mode = mode
        self._reset_maze()
        self.w = len(self.maze[0])
        self.h = len(self.maze)

        self.test_cases = [None] * 1000

        self.teacher = teacher
        self.args = args

    def input_size(self):
        return len(self.maze) * len(self.maze[0])

    def num_actions(self):
        return len(self.action_space)

    def render(self):
        print()
        for r in self.maze:
            print('    ', ''.join(r))

    def reset(self, test_id=None):

        if test_id is not None:
            if self.test_cases[test_id] is None:
                with open('tests/' + self.mode + '/' + str(test_id) + '.json') as f:
                    test_case = json.load(f)
                    self.test_cases[test_id] = test_case
            else:
                test_case = self.test_cases[test_id]
            self.maze = copy.deepcopy(test_case['maze'])
            self.agent_pos = tuple(copy.deepcopy(test_case['agent_pos']))
            self.goal_pos = tuple(copy.deepcopy(test_case['goal_pos']))
        else:
            self._reset_maze()
            self.agent_pos = (0, 0)
            self.maze[self.agent_pos[0]][self.agent_pos[1]] = 'O'
            while True:
                goal_pos = (random.randint(0, self.h - 1), random.randint(0, self.w - 1))
                if goal_pos == self.agent_pos:
                    continue
                if self.maze[goal_pos[0]][goal_pos[1]] != 'x':
                    break
            self.goal_pos = goal_pos
            self.maze[goal_pos[0]][goal_pos[1]] = 'G'

            path = self.shortest_path(self.agent_pos, self.goal_pos)
            for x, y in path:
                if self.maze[x][y] not in ['O', 'G']:
                    self.maze[x][y] = '#'

        if self.args.no_carpet:
            for r in self.maze:
                for i in range(len(r)):
                    if r[i] == '#':
                        r[i] = '.'

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

        if 'hard' in self.mode:
            self.maze = np.tile(self.maze, (2,2)).tolist()

    def _get_ob(self):
        ob = []
        for r in self.maze:
            for c in r:
                ob.append(self.symbol_indices[c])
        return ob

    def shortest_path(self, start_pos, goal_pos):
        prev = []
        for r in self.maze:
            prev.append([None] * len(r))
        queue = [None] * 1000
        start = 0
        end = 0
        queue[end] = start_pos
        end += 1
        prev[start_pos[0]][start_pos[1]] = -1
        while start < end:
            pos = queue[start]
            start += 1
            if pos == goal_pos:
                path = [pos]
                while prev[pos[0]][pos[1]] != -1:
                    pos = prev[pos[0]][pos[1]]
                    path.append(pos)
                return list(reversed(path))

            for i, j in self.action_space:
                new_pos = (pos[0] + i, pos[1] + j)
                if new_pos[0] < 0 or new_pos[0] >= self.h or new_pos[1] < 0 or new_pos[1] >= self.w:
                    continue
                if self.maze[new_pos[0]][new_pos[1]] != 'x' and prev[new_pos[0]][new_pos[1]] == None:
                    queue[end] = new_pos
                    end += 1
                    prev[new_pos[0]][new_pos[1]] = pos

        return None

    def _calculate_valid_actions(self):
        self.valid_action_indices = []
        for k, (i, j) in enumerate(self.action_space):
            new_pos = (self.agent_pos[0] + i, self.agent_pos[1] + j)
            if new_pos[0] < 0 or new_pos[0] >= self.h or new_pos[1] < 0 or new_pos[1] >= self.w:
                continue
            if self.maze[new_pos[0]][new_pos[1]] != 'x':
                self.valid_action_indices.append(k)

    def _get_reward(self, pos):
        if self.maze[pos[0]][pos[1]] == '#':
            reward = 1
        elif self.maze[pos[0]][pos[1]] == 'G':
            reward = 20
        else:
            reward = 0
        return reward

    def step(self, action_idx):
        action = self.action_space[action_idx]
        new_pos = (self.agent_pos[0] + action[0], self.agent_pos[1] + action[1])
        feedback = self.teacher(self, action)

        reward = self._get_reward(new_pos)

        done = self.maze[new_pos[0]][new_pos[1]] == 'G'
        #done = action == (0, 0)

        self.maze[self.agent_pos[0]][self.agent_pos[1]] = '.'
        self.agent_pos = new_pos
        self.maze[self.agent_pos[0]][self.agent_pos[1]] = 'O'

        self._calculate_valid_actions()

        return self._get_ob(), reward, feedback, done
