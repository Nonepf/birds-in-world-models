import torch
import torch.nn as nn
import torch.nn.functional as F

class RSSM(nn.Module):
    def __init__(self, z_dim=50, h_dim=400, action_dim=2, obs_dim=512, min_std=0.1):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.obs_dim = obs_dim
        self.min_std = min_std

        # transition: [z, a] -> mlp -> gru -> mu, sigma
        self.trans_mlp = nn.Sequential(
            nn.Linear(z_dim + action_dim, h_dim),
            nn.ELU(),
        )
        self.gru = nn.GRU(h_dim, h_dim, batch_first=True)
        self.mu_head = nn.Linear(h_dim, z_dim)
        self.sigma_head = nn.Linear(h_dim, z_dim)

        # posterior: [h, o] -> mlp -> mu, sigma
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
        Single-step. obs=None: transition only. obs!=None: + posterior.
        """
        x = torch.cat([prev_z, prev_a], dim=-1)
        x = self.trans_mlp(x)
        x = x.unsqueeze(1)                            # (B, 1, h_dim)
        out, _ = self.gru(x, prev_h.unsqueeze(0).contiguous())     # out: (B, 1, h_dim)
        h = out.squeeze(1)                             # (B, h_dim)
        mu = self.mu_head(h)
        sigma = self.sigma_head(h)
        z_prior = self._sample(mu, sigma)

        if obs is None:
            return h, z_prior, z_prior

        x = torch.cat([h, obs], dim=-1)
        x = self.post_mlp(x)
        mu_post = self.post_mu(x)
        sigma_post = self.post_sigma(x)
        z_post = self._sample(mu_post, sigma_post)
        return h, z_prior, z_post

    def forward_sequence(self, a_seq, o_seq):
        """
        Process full sequence (B, T, dim). Returns h, z_prior, z_post over time.
        GPU: GRU loop is unavoidable (z recurrence), but all MLP heads are
        batched over (B*T) after the loop.
        """
        B, T, _ = a_seq.shape

        h_all = []
        z_prev = torch.zeros(B, self.z_dim, device=a_seq.device)
        h_prev = torch.zeros(1, B, self.h_dim, device=a_seq.device)

        for t in range(T):
            x = torch.cat([z_prev, a_seq[:, t, :]], dim=-1)
            x = self.trans_mlp(x).unsqueeze(1)
            out, h_prev = self.gru(x, h_prev)
            h = out.squeeze(1)
            h_all.append(h)

            # posterior z for next step's transition input
            p_x = torch.cat([h, o_seq[:, t, :]], dim=-1)
            p_x = self.post_mlp(p_x)
            mu_post = self.post_mu(p_x)
            sigma_post = self.post_sigma(p_x)
            z_prev = self._sample(mu_post, sigma_post)

        h_seq = torch.stack(h_all, dim=1)                   # (B, T, h_dim)

        # Batch all heads across (B*T) for GPU efficiency
        h_flat = h_seq.reshape(B * T, self.h_dim)           # (B*T, h_dim)
        o_flat = o_seq.reshape(B * T, self.obs_dim)         # (B*T, obs_dim)

        mu = self.mu_head(h_flat).reshape(B, T, self.z_dim)
        sigma = self.sigma_head(h_flat).reshape(B, T, self.z_dim)
        z_prior = self._sample(mu, sigma)

        p_x = torch.cat([h_flat, o_flat], dim=-1)
        p_x = self.post_mlp(p_x)
        mu_post = self.post_mu(p_x).reshape(B, T, self.z_dim)
        sigma_post = self.post_sigma(p_x).reshape(B, T, self.z_dim)
        z_post = self._sample(mu_post, sigma_post)

        return h_seq, z_prior, z_post, mu, sigma, mu_post, sigma_post
