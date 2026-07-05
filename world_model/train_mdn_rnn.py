import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.distributions as D
import numpy as np

from models import MDNRNN

class RNNTrajectoryDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.z_seqs = data["z_seqs"]
        self.action_seqs = data["action_seqs"]
        
    def __len__(self):
        return len(self.z_seqs)
    
    def __getitem__(self, idx):
        z = torch.tensor(self.z_seqs[idx], dtype=torch.float32)
        
        inputs_z = z[:-1]
        targets_z = z[1:]
        actions = torch.tensor(self.action_seqs[idx][:-1], dtype=torch.long)
        
        return inputs_z, actions, targets_z

# Collate function to dynamic-pad variable-length episodes into a single batch.
def collate_fn(batch):
    inputs_z, actions, targets_z = zip(*batch)
    lengths = torch.tensor([len(x) for x in inputs_z])
    
    pad_inputs_z = pad_sequence(inputs_z, batch_first=True, padding_value=0.0) # type: ignore
    pad_actions = pad_sequence(actions, batch_first=True, padding_value=0) # type: ignore
    pad_targets_z = pad_sequence(targets_z, batch_first=True, padding_value=0.0) # type: ignore
    
    return pad_inputs_z, pad_actions, pad_targets_z, lengths

def mdn_loss_fn(pi, mu, sigma, y, lengths):
    B, T, K, Z = mu.shape
    
    y_ex = y.unsqueeze(2).expand_as(mu)
    
    dist = D.Normal(mu, sigma)
    log_probs = dist.log_prob(y_ex)
    log_probs = torch.sum(log_probs, dim=-1)
    
    log_pi = torch.log(pi + 1e-8)
    log_likelihood = torch.logsumexp(log_pi + log_probs, dim=-1) # (B, T)
    
    mask = torch.zeros(B, T, device=y.device)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1.0
        
    nll_loss = -log_likelihood * mask
    return nll_loss.sum() / mask.sum()

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = RNNTrajectoryDataset("./data/flappy_bird_rnn_dataset.npz")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    model = MDNRNN(z_dim=512, action_dim=2, hidden_dim=256, num_gaussians=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs_z, actions, targets_z, lengths in dataloader:
            inputs_z, actions, targets_z = inputs_z.to(device), actions.to(device), targets_z.to(device)
            
            optimizer.zero_grad()
            
            pi, mu, sigma, _ = model(inputs_z, actions)
            
            loss = mdn_loss_fn(pi, mu, sigma, targets_z, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * lengths.sum().item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Trajectory NLL Loss: {total_loss / lengths.sum().item():.4f}")
        
    torch.save(model.state_dict(), "./checkpoints/mdn_rnn_world_model.pth")
    print("MDN-RNN saved to ./checkpoints/mdn_rnn_world_model.pth")

if __name__ == "__main__":
    run_training()