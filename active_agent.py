import os
import sys

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


from agent import Agent

class ActiveAgent(Agent):

    def __init__(self, model, teacher, optimizer, hparams):

        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.hparams = hparams
        self.query_policy_type = hparams.query_policy_type

    def train(self, env):
        self.model.train()

        # Reset environment
        ob = env.reset()
        self.teacher.reset()

        train_infos = []

        debug = False

        for _ in range(env.w + env.h):

            ob = torch.tensor(ob, dtype=torch.float)

            # Compute mean exe distribution
            exe_dist = self.model(env, self.teacher, ob, env.valid_action_indices, apply_noise=5)

            # Make query decision
            query_logit, query_action = self.model.should_query(
                torch.cat([ob, exe_dist], dim=0), env, self.query_policy_type)
            if query_action == 0:
                # Query learned execution policy
                exe_action = Categorical(exe_dist).sample().item()
                # Teacher returns nothing
                feedback = { 'identity': -1, 'action' : -1, 'distance' : None }
                id_logit = exe_logit = None
            else:
                # Query teacher execution policy
                feedback = self.teacher(env)

                if self.query_policy_type == 'dagger':
                    exe_action = Categorical(exe_dist).sample().item()
                else:
                    exe_action = feedback['action']

                persona_id = torch.tensor(feedback['identity'], dtype=torch.long)
                self.model.exe_policy.sample_noise()
                id_logit, exe_logit = self.model.exe_policy(
                    ob, env.valid_action_indices, persona_id=persona_id, apply_noise=True)

            if self.query_policy_type in ['apil', 'errpred']:
                query_dist = query_logit.softmax(dim=0)
            else:
                query_dist = torch.zeros(2)

            env.record(
                    query_action,
                    self.model.uncertainty,
                    query_dist.tolist()[1],
                    exe_dist.tolist()[0])

            # Take action and receive new observation
            ob, done = env.step(exe_action)

            # Store information
            train_infos.append({
                    'ob'          : ob,
                    'exe_logit'   : exe_logit,
                    'id_logit'    : id_logit,
                    'query_logit' : query_logit,
                    'feedback'    : feedback,
                    'exe_action'  : exe_action,
                    'query_action': query_action,
                    'intrinsic'   : self.model.uncertainty['intrinsic'],
                    'extrinsic'   : self.model.uncertainty['extrinsic'],
                    'model'       : self.model.uncertainty['model']
                })

            if done or not env.valid_action_indices or exe_action == -1:
                break

        return train_infos, done


class ActiveImitateAgent(ActiveAgent):

    SIGMA = 2

    def __init__(self, *args):
        super(ActiveImitateAgent, self).__init__(*args)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    def _long_tensor_wrap(self, data):
        return torch.tensor(data, dtype=torch.long).unsqueeze(0)

    def train(self, env):

        train_infos, done = super(ActiveImitateAgent, self).train(env)

        self.exe_loss = torch.tensor(0.)
        self.id_loss = torch.tensor(0.)
        self.query_loss = torch.tensor(0.)

        min_d = env.d[env.agent_pos]
        progress = (min_d == 0)

        num_queries = 0
        total_intrinsic = 0
        total_extrinsic = 0
        total_model = 0

        for info in reversed(train_infos):

            num_queries += info['query_action']
            total_intrinsic += info['intrinsic']
            total_extrinsic += info['extrinsic']
            total_model += info['model']

            # exe loss
            exe_ref_action = info['feedback']['action']
            if exe_ref_action != -1:
                exe_ref_action = self._long_tensor_wrap(exe_ref_action)
                exe_logit = info['exe_logit'].unsqueeze(0)
                exe_loss = self.loss_fn(exe_logit, exe_ref_action)
                self.exe_loss += exe_loss

                # id loss
                id_ref_action = info['feedback']['identity']
                id_ref_action = self._long_tensor_wrap(id_ref_action)
                id_logit = info['id_logit'].unsqueeze(0)
                self.id_loss += self.loss_fn(id_logit, id_ref_action)

            if self.query_policy_type == 'apil':
                # query loss
                d = info['feedback']['distance']
                if d is not None:
                    progress |= (d >= min_d * self.SIGMA)
                    min_d = min(min_d, d)

                if progress:
                    query_ref_action = 0
                else:
                    query_ref_action = 1
                query_ref_action = self._long_tensor_wrap(query_ref_action)
                query_logit = info['query_logit'].unsqueeze(0)
                self.query_loss += self.loss_fn(query_logit, query_ref_action)

            elif self.query_policy_type == 'errpred':
                exe_ref_action = info['feedback']['action']
                if exe_ref_action != -1:
                    query_logit = info['query_logit'].unsqueeze(0)
                    exe_dist = info['exe_logit'].softmax(dim=0).tolist()
                    query_ref_action = 1 - exe_dist[exe_ref_action] > self.model.threshold
                    query_ref_action = self._long_tensor_wrap(query_ref_action)
                    self.query_loss += self.loss_fn(query_logit, query_ref_action)

        self.loss = self.exe_loss + self.id_loss + self.query_loss

        if self.loss.item() > 1e-8:
            # Update model
            self.update()

        ep_info = {
            'done': done,
            'loss': self.loss.item(),
            'num_queries' : num_queries,
            'total_steps': len(train_infos),
            'intrinsic': total_intrinsic,
            'extrinsic': total_extrinsic,
            'model': total_model,
        }

        return ep_info




