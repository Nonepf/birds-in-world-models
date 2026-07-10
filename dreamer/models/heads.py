import torch
import torch.nn as nn


class ObsDecoder(nn.Module):
    def __init__(self, h_dim, z_dim, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, obs_dim),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))


class RewardPredictor(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))


class Actor(nn.Module):
    def __init__(self, h_dim, z_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, action_dim),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))


class Critic(nn.Module):
    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + z_dim, h_dim), nn.ELU(),
            nn.Linear(h_dim, 1),
        )

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1))
