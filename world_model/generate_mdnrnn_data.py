import os
import numpy as np
import torch
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from gymnasium.wrappers import AddRenderObservation
from models import Encoder 

def collect_rnn_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim=512).to(device)
    encoder.load_state_dict(torch.load("../checkpoints/vae_encoder.pth", map_location=device))
    encoder.eval()

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)

    model = PPO.load("../CnnPolicy/flappy_bird_cnn_model")

    max_episodes = 500
    
    all_z_seqs = []
    all_action_seqs = []

    print(f"Generating MDN-RNN sequential data ({max_episodes} episodes)...")
    obs, info = env.reset()

    current_ep_z = []
    current_ep_actions = []

    while len(all_z_seqs) < max_episodes:
        action, _ = model.predict(obs, deterministic=False)
        
        frame_float = obs.transpose(2, 0, 1).astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_float).unsqueeze(0).to(device) # (1, 3, 512, 288)
        
        with torch.no_grad():
            mu, _ = encoder(frame_tensor)
            z_t = mu.squeeze(0).cpu().numpy()

        current_ep_z.append(z_t)
        current_ep_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            all_z_seqs.append(np.array(current_ep_z, dtype=np.float32))        # (T, 512)
            all_action_seqs.append(np.array(current_ep_actions, dtype=np.int64)) # (T,)
            
            current_ep_z = []
            current_ep_actions = []
            
            if len(all_z_seqs) % 50 == 0:
                print(f"Process: {len(all_z_seqs)} / {max_episodes} episodes collected.")
            
            obs, info = env.reset()

    env.close()

    np.savez_compressed(
        "./data/flappy_bird_rnn_dataset.npz", 
        z_seqs=np.array(all_z_seqs, dtype=object),
        action_seqs=np.array(all_action_seqs, dtype=object)
    )
    print("Saved to ./data/flappy_bird_rnn_dataset.npz")

if __name__ == "__main__":
    collect_rnn_data()