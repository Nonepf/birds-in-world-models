import torch
import torch.nn as nn
import torch.nn.functional as F

class MDNRNN(nn.Module):
    def __init__(self, z_dim=512, action_dim=2, hidden_dim=256, num_gaussians=5):
        """
        MDN-RNN
        """
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians

        # Increase the contribution of the action info to the model decision.
        self.action_embed = nn.Embedding(action_dim, 16)
        
        self.lstm = nn.LSTM(
            input_size=z_dim + 16, 
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # predict pi, mu and sigma
        self.fc_pi = nn.Linear(hidden_dim, num_gaussians)
        self.fc_mu = nn.Linear(hidden_dim, num_gaussians * z_dim)
        self.fc_sigma = nn.Linear(hidden_dim, num_gaussians * z_dim)

    def forward(self, z_seq, action_seq, hidden=None):
        B, T, _ = z_seq.shape
        
        act_embed = self.action_embed(action_seq)          # (B, T, 16)
        lstm_input = torch.cat([z_seq, act_embed], dim=-1) # (B, T, 528)
        
        output, hidden = self.lstm(lstm_input, hidden)     # (B, T, 256)
        
        pi = self.fc_pi(output)                            # (B, T, K)
        pi = F.softmax(pi, dim=-1)                         # sum = 1
        
        mu = self.fc_mu(output)
        mu = mu.view(B, T, self.num_gaussians, self.z_dim) # (B, T, K, 512)
        
        sigma = self.fc_sigma(output)
        sigma = sigma.view(B, T, self.num_gaussians, self.z_dim) # (B, T, K, 512)
        sigma = torch.exp(sigma) # ensure sigma > 0
        
        return pi, mu, sigma, hidden