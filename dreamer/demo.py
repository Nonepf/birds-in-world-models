"""
Play Flappy Bird with trained Dreamer agent.
"""
import os
import sys
import numpy as np
import torch, torch.nn.functional as F
import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium.wrappers import AddRenderObservation, HumanRendering

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from world_model.models import Encoder as VAEEncoder
from models import RSSM, Actor


def encode_frame(obs, vae, device):
    frame = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32) / 255.0)
    frame = frame.unsqueeze(0).to(device)
    o, _ = vae(frame)
    return o


def play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = VAEEncoder(latent_dim=512).to(device)
    vae.load_state_dict(torch.load("../world_model/checkpoints/vae_encoder.pth", map_location=device))
    vae.eval()

    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    rssm.load_state_dict(torch.load("checkpoints/rssm.pth", map_location=device))
    rssm.eval()

    actor = Actor(400, 50, 2).to(device)
    actor.load_state_dict(torch.load("checkpoints/actor.pth", map_location=device))
    actor.eval()

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)
    env = HumanRendering(env)

    obs, _ = env.reset()
    h = torch.zeros(1, 400, device=device)
    z = torch.zeros(1, 50, device=device)
    prev_action = torch.zeros(1, 2, device=device)

    print("Playing... (close window to exit)")
    with torch.no_grad():
        while True:
            o = encode_frame(obs, vae, device)

            h, _, z = rssm(z, prev_action, h, o)

            action_logits = actor(h, z)
            action = torch.argmax(action_logits, dim=1).item()

            prev_action = F.one_hot(torch.tensor([action]), num_classes=2).float().to(device)

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if terminated or truncated:
                print(f"Score: {info.get('score', 0)}")
                obs, _ = env.reset()
                h = torch.zeros(1, 400, device=device)
                z = torch.zeros(1, 50, device=device)
                prev_action = torch.zeros(1, 2, device=device)


if __name__ == "__main__":
    play()
