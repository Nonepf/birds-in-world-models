"""
Generate Dreamer training data (CNN policy + random actions).
"""
import os, sys
import torch
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium.wrappers import AddRenderObservation
from stable_baselines3 import PPO

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from world_model.models import Encoder as VAEEncoder


def collect_episodes(vae_encoder, device, episodes, policy="cnn"):
    """Collect episodes with CNN policy or random actions."""
    if policy == "cnn":
        cnn = PPO.load("../CnnPolicy/flappy_bird_cnn_model")

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)

    all_o, all_a, all_r = [], [], []
    cur_o, cur_a, cur_r = [], [], []

    print(f"Collecting {episodes} episodes (policy={policy})...")
    obs, _ = env.reset()

    while len(all_o) < episodes:
        if policy == "cnn":
            action, _ = cnn.predict(obs, deterministic=False)
        else:
            action = np.random.randint(0, 2)

        frame = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32) / 255.0)
        frame = frame.unsqueeze(0).to(device)
        with torch.no_grad():
            o, _ = vae_encoder(frame)
            o = o.squeeze(0).cpu().numpy()
        cur_o.append(o)
        cur_a.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        cur_r.append(reward)

        if terminated or truncated:
            frame = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32) / 255.0)
            frame = frame.unsqueeze(0).to(device)
            with torch.no_grad():
                o_last, _ = vae_encoder(frame)
                o_last = o_last.squeeze(0).cpu().numpy()
            cur_o.append(o_last)
            cur_a.append(0)
            cur_r.append(0.0)

            all_o.append(np.array(cur_o, dtype=np.float32))
            all_a.append(np.array(cur_a, dtype=np.int64))
            all_r.append(np.array(cur_r, dtype=np.float32))
            cur_o, cur_a, cur_r = [], [], []

            if len(all_o) % 50 == 0:
                print(f"  {len(all_o)}/{episodes} episodes")
            obs, _ = env.reset()

    env.close()
    return all_o, all_a, all_r


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = VAEEncoder(latent_dim=512).to(device)
    vae.load_state_dict(torch.load("../world_model/checkpoints/vae_encoder.pth", map_location=device))
    vae.eval()

    # Collect CNN + random data, then merge
    cnn_o, cnn_a, cnn_r = collect_episodes(vae, device, 300, policy="cnn")
    rand_o, rand_a, rand_r = collect_episodes(vae, device, 300, policy="random")

    all_o = cnn_o + rand_o
    all_a = cnn_a + rand_a
    all_r = cnn_r + rand_r

    save_path = "data/dreamer_trajectories.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path,
                         o_seqs=np.array(all_o, dtype=object),
                         a_seqs=np.array(all_a, dtype=object),
                         r_seqs=np.array(all_r, dtype=object))
    print(f"Saved {len(all_o)} episodes ({len(cnn_o)} CNN + {len(rand_o)} random) to {save_path}")


if __name__ == "__main__":
    main()
