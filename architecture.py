import os
import sys
from scipy.stats import entropy as compute_entropy

import torch
import torch.nn as nn
import torch.distributions as D
import torch.functional as F
from torch.distributions.categorical import Categorical

from concrete_dropout import *


HIDDEN_SIZE=100
PERSONA_SIZE=HIDDEN_SIZE // 2
K=2


class IdentityModel(nn.Module):

    def __init__(self, input_size, hidden_size=HIDDEN_SIZE):

        super(IdentityModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, K),
        )

    def __call__(self, state, apply_noise=False):
        logit = self.model(state)
        return logit


class ExecutionPolicy(nn.Module):

    def __init__(self, input_size, num_actions, name, hidden_size=HIDDEN_SIZE,
            persona_size=PERSONA_SIZE):

        super(ExecutionPolicy, self).__init__()
        self.identity_model = IdentityModel(input_size)
        self.persona_embed = nn.Embedding(K, persona_size)
        self.policy_model = nn.Sequential(
            nn.Linear(input_size + persona_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.input_size = input_size
        self.persona_size = persona_size

        self.dropout_rate = 0.2

    def sample_noise(self):
        input_size = self.input_size + self.persona_size
        self.noise = D.Bernoulli(
            torch.tensor([self.dropout_rate] * input_size)).sample()

    def _mask_out_invalid_actions(self, logit, valid_indices):
        for idx in range(logit.shape[-1]):
            if idx not in valid_indices:
                logit[idx] = -float('inf')

    def __call__(self, state, valid_action_indices, persona_id=None, apply_noise=False):

        identity_logit = self.identity_model(state)
        if persona_id is None:
            persona_id = D.Categorical(logits=identity_logit).sample()
        persona_embedding = self.persona_embed(persona_id)
        policy_input = torch.cat([state, persona_embedding])

        if apply_noise:
            policy_input = policy_input * self.noise / self.dropout_rate

        policy_logit = self.policy_model(policy_input)
        self._mask_out_invalid_actions(policy_logit, valid_action_indices)

        return identity_logit, policy_logit


class QueryPolicy(nn.Module):

    def __init__(self, input_size, name, hidden_size=HIDDEN_SIZE):

        super(QueryPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def __call__(self, state):
        logit = self.model(state)
        dist = logit.softmax(dim=0)
        return logit, D.Categorical(dist).sample().item()


class ActiveImitateModel(nn.Module):

    def __init__(self, input_size, num_actions, hparams):

        super(ActiveImitateModel, self).__init__()
        self.exe_policy = ExecutionPolicy(input_size, num_actions, 'exe')
        self.query_policy = QueryPolicy(input_size + num_actions, 'query')
        self.threshold = hparams.threshold
        self.compute_efficiency = lambda x: compute_entropy(x, base=len(x))
        self.n_samples = hparams.n_samples

    def mutual_information_decomposition(self, dists):
        dists = [d.numpy() for d in dists]
        total_entropy = self.compute_efficiency(np.mean(dists, axis=0))
        conditional_entropy = np.mean([self.compute_efficiency(d) for d in dists])
        mutual_information = total_entropy - conditional_entropy

        return total_entropy, conditional_entropy, mutual_information


    def sample_policy(self, state, valid_action_indices, apply_noise=False):

        # Draw samples
        dists = []
        for _ in range(self.n_samples):
            id_logit, logit = self.exe_policy(
                state, valid_action_indices, apply_noise=apply_noise)
            dists.append(logit.softmax(dim=0).detach())

        behavioral_entropy, intrinsic_entropy, extrinsic_entropy = \
            self.mutual_information_decomposition(dists)

        self.uncertainty = { 'intrinsic': intrinsic_entropy,
                             'extrinsic': extrinsic_entropy,
                             'model': -1 }
        return sum(dists) / len(dists)

    def __call__(self, env, teacher, state, valid_action_indices, apply_noise=None):

        if apply_noise is None:
            return self.sample_policy(state, valid_action_indices)

        intrinsic_uncertainties = []
        extrinsic_uncertainties = []
        mean_dists = []
        for _ in range(apply_noise):
            self.exe_policy.sample_noise()
            mean_dist = self.sample_policy(state, valid_action_indices, apply_noise=True)
            mean_dists.append(mean_dist)
            intrinsic_uncertainties.append(self.uncertainty['intrinsic'])
            extrinsic_uncertainties.append(self.uncertainty['extrinsic'])

        _, _, model_uncertainty = self.mutual_information_decomposition(mean_dists)

        self.uncertainty = { 'intrinsic': sum(intrinsic_uncertainties) / apply_noise,
                             'extrinsic': sum(extrinsic_uncertainties) / apply_noise,
                             'model': model_uncertainty
                           }

        return sum(mean_dists) / len(mean_dists)


    def should_query(self, state, env, query_policy_type):
        if query_policy_type in ['bc', 'dagger']:
            return None, 1
        elif query_policy_type == 'inuncrty':
            return None, self.uncertainty['intrinsic'] > self.threshold
        elif query_policy_type == 'exuncrty':
            print(env.agent_pos, self.uncertainty['extrinsic'])
            return None, self.uncertainty['extrinsic'] > self.threshold

        assert query_policy_type in ['apil', 'errpred']
        return self.query_policy(state)

