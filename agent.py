import os
import sys

import torch
import torch.nn as nn

class Agent(object):

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, env):
        self.model.train()

        ob = env.reset()
        #env.render()

        best_reward = len(env.shortest_path(env.agent_pos, env.goal_pos)) + 20 - 2

        train_infos = []

        for _ in range(env.w + env.h):
            logit = self.model(ob)
            self._mask_out_invalid_actions(logit, env)
            dist = logit.softmax(dim=0)
            m = torch.distributions.categorical.Categorical(dist)
            action = m.sample()
            ob, reward, feedback, done = env.step(action)

            #env.render()


            train_infos.append({
                    'ob'      : ob,
                    'logit'   : logit,
                    'reward'  : reward,
                    'feedback': feedback,
                    'action'  : action
                })

            if done:
                break

        return train_infos, best_reward, done

    def test(self, env, test_id):
        self.model.eval()

        ob = env.reset(test_id=test_id)

        best_reward = len(env.shortest_path(env.agent_pos, env.goal_pos)) + 20 - 2
        total_reward = 0
        feedbacks = []
        with torch.no_grad():
            for _ in range(env.w + env.h):
                logit = self.model(ob)
                self._mask_out_invalid_actions(logit, env)
                action = logit.argmax(dim=0)
                ob, reward, feedback, done = env.step(action)
                total_reward += reward
                if done:
                    return 1, total_reward, best_reward
        return 0, total_reward, best_reward

    def update(self):
        self.model.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def _mask_out_invalid_actions(self, logit, env):
        for idx in range(len(env.action_space)):
            if idx not in env.valid_action_indices:
                logit[idx] = -float('inf')


class ReinforceAgent(Agent):

    def __init__(self, model, optimizer):
        super(ReinforceAgent, self).__init__(model, optimizer)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')

    def train(self, env):

        train_infos, best_reward, done = super(ReinforceAgent, self).train(env)
        agent_reward = 0
        cum_reward = 0
        actor_loss = 0
        critic_loss = 0
        for info in reversed(train_infos):

            agent_reward += info['reward']

            cum_reward += info['feedback']
            baseline_reward = self.model.predict_baseline(info['ob'])
            norm_reward = cum_reward - baseline_reward

            actor_loss += self.loss_fn(info['logit'].unsqueeze(0), info['action'].unsqueeze(0)) * \
                norm_reward.detach()
            critic_loss += norm_reward ** 2

        self.loss = (actor_loss + critic_loss) / len(train_infos)

        self.update()

        ep_info = {
            'done': done,
            'best_reward': best_reward,
            'agent_reward': agent_reward,
            'loss': self.loss.item(),
            'actor_loss': actor_loss.item() / len(train_infos),
            'critic_loss': critic_loss.item() / len(train_infos)
        }

        return ep_info


class ImitateAgent(Agent):

    def __init__(self, model, optimizer):
        super(ImitateAgent, self).__init__(model, optimizer)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self, env):

        train_infos, best_reward, done = super(ImitateAgent, self).train(env)

        self.loss = 0
        agent_reward = 0
        for info in train_infos:

            agent_reward += info['reward']

            logit = info['logit'].unsqueeze(0)
            reference_action = torch.tensor(info['feedback'], dtype=torch.long).unsqueeze(0)
            self.loss += self.loss_fn(logit, reference_action)

        self.loss = self.loss / len(train_infos)

        self.update()

        ep_info = {
            'done': done,
            'best_reward': best_reward,
            'agent_reward': agent_reward,
            'loss': self.loss.item(),
        }

        return ep_info




