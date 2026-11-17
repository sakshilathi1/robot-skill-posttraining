"""Small MLP policy: observation -> normalised joint delta."""

import numpy as np
import torch
import torch.nn as nn

from . import config as C


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim=C.OBS_DIM, act_dim=C.ACT_DIM, hidden=C.HIDDEN):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, act_dim))
        self.net = nn.Sequential(*layers)
        # Observation normaliser, frozen after baseline training so that every
        # post-training arm sees identical input preprocessing.
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_normalizer(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def forward(self, obs):
        return self.net((obs - self.obs_mean) / self.obs_std)

    @torch.no_grad()
    def act(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return self.forward(o).squeeze(0).numpy()


def save(model, path):
    torch.save(model.state_dict(), path)


def load(path):
    m = MLPPolicy()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m
