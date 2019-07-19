import os
import sys
import json
import random

import env
import teacher
import flags

random.seed(100)

parser = flags.make_parser()
args = parser.parse_args()

teacher = teacher.RewardTeacher()
env = env.Environment('test_hard', teacher, args)

for i in range(1000):
    env.reset()
    test_case = {
        'maze': env.maze,
        'agent_pos': env.agent_pos,
        'goal_pos': env.goal_pos
    }
    with open('tests/test_hard/' + str(i) + '.json', 'w') as f:
        json.dump(test_case, f, indent=2)
