"""
Train the world model (RSSM + ObsDecoder + RewardPredictor) with ELBO.
MSE on VAE latent is used instead of log-prob on pixels (accepted deviation
due to frozen VAE — see README).
"""
import torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from models import RSSM, ObsDecoder, RewardPredictor


class DreamerDataset(Dataset):
    def __init__(self, episodes):
        """episodes: list of (o_seq, a_seq, r_seq) numpy arrays, or path to npz."""
        if isinstance(episodes, str):
            data = np.load(episodes, allow_pickle=True)
            self.o_seqs = list(data["o_seqs"])
            self.a_seqs = list(data["a_seqs"])
            self.r_seqs = list(data["r_seqs"])
        else:
            self.o_seqs = [e[0] for e in episodes]
            self.a_seqs = [e[1] for e in episodes]
            self.r_seqs = [e[2] for e in episodes]

    def __len__(self):
        return len(self.o_seqs)

    def __getitem__(self, idx):
        o = torch.from_numpy(self.o_seqs[idx].astype(np.float32))
        a = torch.from_numpy(self.a_seqs[idx].astype(np.int64))
        r = torch.from_numpy(self.r_seqs[idx].astype(np.float32))
        return o[:-1], a[:-1], r[:-1], o[1:]


def collate_fn(batch):
    o_in, a, r, o_next = zip(*batch)
    lengths = torch.tensor([len(x) for x in o_in])
    o_in = pad_sequence(o_in, batch_first=True) #type:ignore
    a = pad_sequence(a, batch_first=True) #type:ignore
    r = pad_sequence(r, batch_first=True) #type:ignore
    o_next = pad_sequence(o_next, batch_first=True) #type:ignore
    return o_in, a, r, o_next, lengths


def kl_divergence(mu_post, sigma_post, mu_prior, sigma_prior, min_std):
    """KL(N(μ_q,σ_q) || N(μ_p,σ_p)), summed over z_dim. Inputs are in softplus space."""
    std_q = F.softplus(sigma_post) + min_std
    std_p = F.softplus(sigma_prior) + min_std
    var_q = std_q ** 2
    var_p = std_p ** 2
    kl = 0.5 * (torch.log(var_p / var_q) + (var_q + (mu_post - mu_prior) ** 2) / var_p - 1)
    return kl.sum(dim=-1)  # (B, T)


def train_world_model(rssm, obs_dec, reward_pred, dataset, device, epochs=1, steps=100):
    """Train for `steps` gradient steps. Returns loss dict for logging."""
    loader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=collate_fn)
    params = list(rssm.parameters()) + list(obs_dec.parameters()) + list(reward_pred.parameters())
    opt = optim.Adam(params, lr=6e-4)

    rssm.train(); obs_dec.train(); reward_pred.train()
    total_obs, total_rew, total_kl, total_steps = 0, 0, 0, 0
    step = 0

    for epoch in range(epochs):
        for o_in, a, r, o_next, lengths in loader:
            if step >= steps:
                break
            o_in = o_in.to(device); a = a.to(device)
            r = r.to(device); o_next = o_next.to(device)
            lengths = lengths.to(device)
            B, T, _ = o_in.shape

            opt.zero_grad()

            a_onehot = F.one_hot(a, num_classes=2).float()
            h_seq, z_prior, z_post, mu_p, sigma_p, mu_q, sigma_q = \
                rssm.forward_sequence(a_onehot, o_in)

            kl = kl_divergence(mu_q, sigma_q, mu_p, sigma_p, rssm.min_std)  # (B, T)
            kl = torch.clamp(kl, min=3.0)  # free nats

            h_flat = h_seq.reshape(B * T, 400)
            z_flat = z_post.reshape(B * T, 50)
            o_flat = o_next.reshape(B * T, 512)
            r_flat = r.reshape(B * T)
            kl_flat = kl.reshape(B * T)

            o_pred = obs_dec(h_flat, z_flat)
            r_pred = reward_pred(h_flat, z_flat).squeeze(-1)

            mask = (torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)).reshape(B * T)

            obs_loss = (F.mse_loss(o_pred, o_flat, reduction='none').mean(dim=-1) * mask).sum()
            rew_loss = (F.mse_loss(r_pred, r_flat, reduction='none') * mask).sum()
            kl_loss = (kl_flat * mask).sum()
            valid = mask.sum().clamp(min=1)

            loss = (obs_loss + rew_loss + kl_loss) / valid
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()

            total_obs += obs_loss.item()
            total_rew += rew_loss.item()
            total_kl += kl_loss.item()
            total_steps += valid.item()
            step += 1

    return {
        "obs": total_obs / max(total_steps, 1),
        "rew": total_rew / max(total_steps, 1),
        "kl": total_kl / max(total_steps, 1),
        "steps": step,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DreamerDataset("data/dreamer_trajectories.npz")
    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    obs_dec = ObsDecoder(400, 50, 512).to(device)
    reward_pred = RewardPredictor(400, 50).to(device)

    for epoch in range(30):
        info = train_world_model(rssm, obs_dec, reward_pred, dataset, device, epochs=1, steps=200)
        print(f"Epoch [{epoch+1:2d}/30] obs={info['obs']:.4f}  rew={info['rew']:.4f}  "
              f"kl={info['kl']:.4f}")

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(rssm.state_dict(), "checkpoints/rssm.pth")
    torch.save(obs_dec.state_dict(), "checkpoints/obs_decoder.pth")
    torch.save(reward_pred.state_dict(), "checkpoints/reward_predictor.pth")
    print("Saved to checkpoints/")


if __name__ == "__main__":
    main()
