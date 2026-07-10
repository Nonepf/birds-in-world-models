import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, z_dim=50, h_dim=400, action_dim=2, obs_dim=512, min_std=0.1):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.min_std = min_std

        # transition: [z, a] -> gru -> mu, sigma
        self.trans_mlp = nn.Sequential(
            nn.Linear(z_dim + action_dim, h_dim),
            nn.ELU(),
        )
        self.gru = nn.GRUCell(h_dim, h_dim)
        self.mu_head = nn.Linear(h_dim, z_dim)
        self.sigma_head = nn.Linear(h_dim, z_dim)

        # posterior: [h, o] -> mu, sigma
        self.post_mlp = nn.Sequential(
            nn.Linear(h_dim + obs_dim, h_dim),
            nn.ELU(),
        )
        self.post_mu = nn.Linear(h_dim, z_dim)
        self.post_sigma = nn.Linear(h_dim, z_dim)

    def _sample(self, mu, sigma):
        eps = torch.randn_like(mu)
        return mu + F.softplus(sigma) * eps + self.min_std

    def forward(self, prev_z, prev_a, prev_h, obs=None):
        """
        obs  = None: transition
        obs != None: transition + posterior(representation)
        """
        # transition
        x = torch.cat([prev_z, prev_a], dim=-1)
        x = self.trans_mlp(x)
        h = self.gru(x, prev_h)
        mu = self.mu_head(h)
        sigma = self.sigma_head(h)
        z_prior = self._sample(mu, sigma)

        # posterior
        if obs is None:
            return h, z_prior, z_prior

        x = torch.cat([h, obs], dim=-1)
        x = self.post_mlp(x)
        mu_post = self.post_mu(x)
        sigma_post = self.post_sigma(x)
        z_post = self._sample(mu_post, sigma_post)
        return h, z_prior, z_post
