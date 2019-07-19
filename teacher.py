import os
import sys

class Teacher(object):

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

    def __call__(self, env, action):
        # Find closes '#'
        maze = env.maze
        best_dist = 1e9
        best_dest = None

        pos = env.agent_pos

        for i, r in enumerate(maze):
            for j, c in enumerate(r):
                if c == '#':
                    path = env.shortest_path(pos, (i, j))
                    dist = len(path)
                    if dist < best_dist:
                        best_dist = dist
                        best_dest = (i, j)

        if best_dest is None:
            best_dest = env.goal_pos

        path = env.shortest_path(pos, best_dest)
        new_pos = path[1]
        action = (new_pos[0] - pos[0], new_pos[1] - pos[1])

        for idx, a in enumerate(env.action_space):
            if action == a:
                return idx

        return None


    def _is_valid(self, env, pos):
        return pos[0] >= 0 and pos[0] < env.h and pos[1] >= 0 and pos[1] < env.w \
                and env.maze[pos[0]][pos[1]] != 'x'
