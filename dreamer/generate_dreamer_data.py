"""
Generate seed episodes for Dreamer.
"""
import os, sys, torch, numpy as np
import gymnasium as gym, flappy_bird_gymnasium
from gymnasium.wrappers import AddRenderObservation

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from world_model.models import Encoder as VAEEncoder


def collect_random(vae, device, episodes=5):
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)
    all_o, all_a, all_r = [], [], []

    for ep in range(episodes):
        cur_o, cur_a, cur_r = [], [], []
        obs, _ = env.reset()
        t, tr = False, False
        while not (t or tr):
            a = np.random.randint(0, 2)
            fr = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32)/255.0)
            with torch.no_grad():
                o, _ = vae(fr.unsqueeze(0).to(device)); o = o.squeeze(0).cpu().numpy()
            cur_o.append(o); cur_a.append(a)
            obs, r, t, tr, _ = env.step(a); cur_r.append(r)
        fr = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32)/255.0)
        with torch.no_grad():
            o_last, _ = vae(fr.unsqueeze(0).to(device))
            o_last = o_last.squeeze(0).cpu().numpy()
        cur_o.append(o_last); cur_a.append(0); cur_r.append(0.0)
        all_o.append(np.array(cur_o, dtype=np.float32))
        all_a.append(np.array(cur_a, dtype=np.int64))
        all_r.append(np.array(cur_r, dtype=np.float32))
        print(f"  Episode {ep+1}/{episodes}: {len(cur_o)-1} steps, r={sum(cur_r):.1f}")
    env.close()
    return all_o, all_a, all_r


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    vae = VAEEncoder(latent_dim=512).to(device).eval()
    vae.load_state_dict(torch.load("../world_model/checkpoints/vae_encoder.pth", map_location=device))
    all_o, all_a, all_r = collect_random(vae, device, 5)
    os.makedirs("data", exist_ok=True)
    np.savez_compressed("data/dreamer_trajectories.npz",
                         o_seqs=np.array(all_o, dtype=object),
                         a_seqs=np.array(all_a, dtype=object),
                         r_seqs=np.array(all_r, dtype=object))
    print(f"Saved {len(all_o)} seed episodes.")


if __name__ == "__main__":
    main()
