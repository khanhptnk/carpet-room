import os
import sys

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


from agent import Agent

class ActiveAgent(Agent):

    def __init__(self, model, teacher, optimizer):

        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer

    def train(self, env):
        self.model.train()

        # Reset environment
        ob = env.reset()

        best_reward = len(env.shortest_path(env.agent_pos, env.goal_pos)) + 20 - 2

        train_infos = []

        num_asks = 0

        MAX_ASKS = env.distance_to_goal() / 8

        for _ in range(env.w + env.h):

            distance_to_goal = env.distance_to_goal()

            # Predict action distribution based on observation
            exe_logit = self.model(ob)
            self._mask_out_invalid_actions(exe_logit, env)
            exe_dist = exe_logit.softmax(dim=0)

            # Query help-request policy
            ask_logit = self.model.predict_ask(ob + exe_dist.tolist())
            ask_dist = ask_logit.softmax(dim=0)
            ask_action = Categorical(ask_dist).sample().item()

            if num_asks >= MAX_ASKS:
                ask_action = 0
            num_asks += ask_action

            if ask_action == 0:
                # Query the learner's policy
                exe_action = Categorical(exe_dist).sample().item()
                feedback = None
            else:
                # Query the teacher's policy
                exe_action = feedback = self.teacher(env, None)

            # Take action and receive new observation
            ob, reward, feedback, done = env.step(exe_action)

            # Store information
            train_infos.append({
                    'ob'        : ob,
                    'exe_logit' : exe_logit,
                    'ask_logit' : ask_logit,
                    'reward'    : reward,
                    'feedback'  : feedback,
                    'exe_action': exe_action,
                    'ask_action': ask_action,
                    'distance_to_goal': distance_to_goal
                })

            if done:
                break

        return train_infos, best_reward, done


class ActiveImitateAgent(ActiveAgent):

    def __init__(self, *args):
        super(ActiveImitateAgent, self).__init__(*args)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, env):

        train_infos, best_reward, done = \
            super(ActiveImitateAgent, self).train(env)

        self.exe_losses = []
        self.ask_losses = []
        agent_reward = 0

        min_d = env.distance_to_goal()

        num_asks = 0

        for info in reversed(train_infos):

            agent_reward += info['reward']
            num_asks += info['ask_action']

            if info['feedback'] is not None:

                ## EXE LOSS
                exe_logit = info['exe_logit'].unsqueeze(0)
                exe_ref_action = torch.tensor(
                        info['feedback'], dtype=torch.long).unsqueeze(0)
                # IL loss = cross entropy
                self.exe_losses.append(self.loss_fn(exe_logit, exe_ref_action))

            ## ASK LOSS
            if min_d * 2 <= info['distance_to_goal']:
                ask_ref_action = torch.tensor(0,
                    dtype=torch.long).unsqueeze(0)
            else:
                ask_ref_action = torch.tensor(1,
                    dtype=torch.long).unsqueeze(0)
            ask_logit = info['ask_logit'].unsqueeze(0)
            self.ask_losses.append(self.loss_fn(ask_logit, ask_ref_action))

            # Update min_d
            min_d = min(min_d, info['distance_to_goal'])

        self.loss = sum(self.exe_losses) / len(self.exe_losses) + \
            sum(self.ask_losses) / len(self.ask_losses)

        # Update model
        self.update()

        ep_info = {
            'done': done,
            'best_reward': best_reward,
            'agent_reward': agent_reward,
            'loss': self.loss.item(),
            'ask_ratio': num_asks / len(train_infos)
        }

        #print(num_asks, '/', len(train_infos), env.w, ep_info['ask_ratio'])

        return ep_info




