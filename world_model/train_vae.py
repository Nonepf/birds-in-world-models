import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from models import Encoder, Decoder

class FlappyBirdDataset(Dataset):
    def __init__(self, npz_path):
        self.images = np.load(npz_path)["data"] # (N, 3, 512, 288)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        return torch.from_numpy(img)

def vae_loss_fn(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL Loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss, recon_loss, kld_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset = FlappyBirdDataset("./data/flappy_bird_vae_dataset.npz")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)
    
    epochs = 20
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        
        total_loss, total_recon, total_kld = 0, 0, 0
        
        for x in dataloader:
            x = x.to(device) # (B, 3, 512, 288)
            
            optimizer.zero_grad()
            
            mu, logvar = encoder(x)
            z = encoder.reparameterize(mu, logvar)
            recon_x = decoder(z)
            
            loss, recon_loss, kld_loss = vae_loss_fn(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld_loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {total_loss/len(dataset):.2f} | "
              f"Recon Loss: {total_recon/len(dataset):.2f} | KLD: {total_kld/len(dataset):.2f}")

    torch.save(encoder.state_dict(), "./checkpoints/vae_encoder.pth")
    torch.save(decoder.state_dict(), "./checkpoints/vae_decoder.pth")
    print("Finished! Models are saved to ./checkpoints")

if __name__ == "__main__":
    main()