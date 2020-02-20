import os
import sys

import torch
import torch.nn as nn

class LinearModel(nn.Module):

    def __init__(self, input_size, num_actions):

        super(LinearModel, self).__init__()

        self.linear_layer = nn.Linear(input_size, num_actions)

    def __call__(self, ob):
        if isinstance(ob, list):
            ob = torch.tensor(ob, dtype=torch.float)
        logit = self.linear_layer(ob)
        return logit


class MLPModel(nn.Module):

    def __init__(self, input_size, num_actions, hidden_size=100):

        super(MLPModel, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size, num_actions)

    def __call__(self, ob):
        if isinstance(ob, list):
            ob = torch.tensor(ob, dtype=torch.float)
        hidden = self.relu(self.input_layer(ob))
        logit  = self.output_layer(hidden)
        return logit


class ReinforceModel(nn.Module):

    def __init__(self, input_size, num_actions, model_class):

        super(ReinforceModel, self).__init__()

        self.model = model_class(input_size, num_actions)
        self.baseline_model = model_class(input_size, 1)

    def __call__(self, ob):
        return self.model(ob)

    def predict_baseline(self, ob):
        if isinstance(ob, list):
            ob = torch.tensor(ob, dtype=torch.float)
        return self.baseline_model(ob)


class ActiveImitateModel(nn.Module):

    def __init__(self, input_size, num_actions, model_class):

        super(ActiveImitateModel, self).__init__()

        self.exe_model = model_class(input_size, num_actions)
        self.ask_model = model_class(input_size + num_actions, 2)

    def __call__(self, ob):
        return self.exe_model(ob)

    def predict_ask(self, ob):
        return self.ask_model(ob)
