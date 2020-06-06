import os
import sys
import random

class Teacher(object):

    def __init__(self, args):
        self.args = args

    def __call__(self, *args, **kwargs):
        pass


class RewardTeacher(Teacher):

    def __call__(self, env, action):
        pos = (env.agent_pos[0] + action[0], env.agent_pos[1] + action[1])
        cell = env.maze[pos[0]][pos[1]]
        if cell == '#':
            return 1
        if cell == 'G':
            return 20
        return 0


class ActionTeacher(Teacher):

    K = 2

    def __init__(self, teacher_type):
        self.teacher_type = teacher_type
        #self.identity = 0

    def reset(self):
        if 'two' in self.teacher_type:
            self.identity = random.choice(range(self.K))
            #self.identity = 1 - self.identity

    def __call__(self, env):

        agent_pos = env.agent_pos
        d = env.d
        actions = []
        for k, (i, j) in enumerate(env.action_space):
            new_pos = (agent_pos[0] + i, agent_pos[1] + j)
            if k not in env.valid_action_indices:
                continue
            if d[agent_pos] == d[new_pos] + 1:
                actions.append(k)

        if self.teacher_type == 'detm':
            identity = 0
            action = actions[0]
        elif self.teacher_type == 'rand':
            identity = 0
            action = random.choice(actions)
        elif self.teacher_type == 'twodifdetm':
            identity = self.identity
            if identity == 0:
                action = actions[0]
            else:
                assert identity == 1
                action = actions[-1]
        else:
            assert self.teacher_type == 'tworand'
            identity = self.identity
            action = random.choice(actions)

        feedback = { 'identity': identity,
                     'action'  : action,
                     'distance': d[agent_pos] }

        return feedback



