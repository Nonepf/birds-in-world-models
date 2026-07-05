""" controller """

import torch
import torch.nn as nn

class Controller(nn.Module):
    def __init__(self, z_dim=512, hidden_dim=256, action_dim=2):
        super().__init__()
        self.fc = nn.Linear(z_dim + hidden_dim, action_dim)

    def forward(self, *inputs):
        cat_in = torch.cat(inputs, dim=1)
        return self.fc(cat_in)