"""
Train the world model.
"""
import os, sys
import torch, torch.nn.functional as F, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from models import RSSM
from models import ObsDecoder, RewardPredictor

class DreamerDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.o_seqs = data["o_seqs"]
        self.a_seqs = data["a_seqs"]
        self.r_seqs = data["r_seqs"]

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
    o_in = pad_sequence(o_in, batch_first=True) #type: ignore
    a = pad_sequence(a, batch_first=True) #type: ignore
    r = pad_sequence(r, batch_first=True) #type: ignore
    o_next = pad_sequence(o_next, batch_first=True) #type: ignore
    return o_in, a, r, o_next, lengths

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = DreamerDataset("data/dreamer_trajectories.npz")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    obs_dec = ObsDecoder(400, 50, 512).to(device)
    reward_pred = RewardPredictor(400, 50).to(device)

    params = list(rssm.parameters()) + list(obs_dec.parameters()) + list(reward_pred.parameters())
    opt = optim.Adam(params, lr=1e-3)

    for epoch in range(30):
        rssm.train(); obs_dec.train(); reward_pred.train()
        total_obs_loss, total_rew_loss = 0, 0
        total_len = 0

        for o_in, a, r, o_next, lengths in loader:
            o_in = o_in.to(device); a = a.to(device)
            r = r.to(device); o_next = o_next.to(device)
            lengths = lengths.to(device)
            B, T, _ = o_in.shape

            opt.zero_grad()

            h = torch.zeros(B, 400, device=device)
            z = torch.zeros(B, 50, device=device)
            mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)

            obs_loss_sum, rew_loss_sum = 0, 0
            valid_steps = 0

            for t in range(T):
                o_t = o_in[:, t, :]
                a_t = a[:, t]
                o_tp1 = o_next[:, t, :]
                r_t = r[:, t]

                a_onehot = F.one_hot(a_t, num_classes=2).float()

                h, _, z_post = rssm(z, a_onehot, h, o_t)

                o_pred = obs_dec(h, z_post)
                obs_loss_sum += (F.mse_loss(o_pred, o_tp1, reduction='none')
                                 .mean(dim=-1) * mask[:, t]).sum()

                r_pred = reward_pred(h, z_post).squeeze(-1)
                rew_loss_sum += (F.mse_loss(r_pred, r_t, reduction='none')
                                 * mask[:, t]).sum()

                valid_steps += mask[:, t].sum().item()
                z = z_post

            loss = (obs_loss_sum + rew_loss_sum) / max(valid_steps, 1)
            loss.backward() #type: ignore
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()

            total_obs_loss += obs_loss_sum.item() #type: ignore
            total_rew_loss += rew_loss_sum.item() #type: ignore
            total_len += valid_steps

        print(f"Epoch [{epoch+1:2d}/30] obs={total_obs_loss/max(total_len,1):.4f}  "
              f"rew={total_rew_loss/max(total_len,1):.4f}")

    torch.save(rssm.state_dict(), "checkpoints/rssm.pth")
    torch.save(obs_dec.state_dict(), "checkpoints/obs_decoder.pth")
    torch.save(reward_pred.state_dict(), "checkpoints/reward_predictor.pth")
    print("Saved to checkpoints/")

if __name__ == "__main__":
    main()
