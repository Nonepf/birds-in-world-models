"""
Extract features from the display window.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
The original observation space: (height, width, channels) = (512, 288, 3),
which should be converted to (3, 512, 288) manually.
"""

class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = 512
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), # -> (32, 256, 144)
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 128, 72)
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 64, 36)
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # -> (256, 32, 18)
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256 * 32 * 18, latent_dim)
        self.fc_logvar = nn.Linear(256 * 32 * 18, latent_dim)

    def forward(self, X):
        # X: (batch_size, 3, 512, 288)
        X = self.conv_layer(X)
        X_flatten = X.flatten(start_dim=1)

        mu = self.fc_mu(X_flatten)
        logvar = self.fc_logvar(X_flatten)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 32 * 18)
        
        # Upsample + Conv2d
        self.upsample_layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 32, 18)
        x = self.upsample_layers(x)
        return x