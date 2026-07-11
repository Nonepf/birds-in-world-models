"""
Dreamer V1 online training loop.
"""
import os, sys, torch, numpy as np
import gymnasium as gym, flappy_bird_gymnasium
from gymnasium.wrappers import AddRenderObservation

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from world_model.models import Encoder as VAEEncoder
from models import RSSM, ObsDecoder, RewardPredictor, Actor, Critic
from train_world_model import DreamerDataset, train_world_model
from train_ac import train_ac


def collect_episode(vae, rssm, actor, device):
    """Play one episode with actor (sample actions for exploration)."""
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)

    obs, _ = env.reset()
    h = torch.zeros(1, 400, device=device)
    z = torch.zeros(1, 50, device=device)
    pa = torch.zeros(1, 2, device=device)
    cur_o, cur_a, cur_r = [], [], []

    with torch.no_grad():
        while True:
            fr = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32)/255.0)
            o, _ = vae(fr.unsqueeze(0).to(device))
            h, _, z = rssm(z, pa, h, o)
            logits = actor(h, z)
            a = torch.distributions.Categorical(logits=logits).sample().item()
            cur_o.append(o.squeeze(0).cpu().numpy()); cur_a.append(a)
            obs, r, t, tr, _ = env.step(a); cur_r.append(r)
            pa = torch.nn.functional.one_hot(torch.tensor([a]), 2).float().to(device)
            if t or tr: break

        fr = torch.from_numpy(obs.transpose(2, 0, 1).astype(np.float32)/255.0)
        o_last, _ = vae(fr.unsqueeze(0).to(device))
        cur_o.append(o_last.squeeze(0).cpu().numpy()); cur_a.append(0); cur_r.append(0.0)

    env.close()
    return (np.array(cur_o, dtype=np.float32),
            np.array(cur_a, dtype=np.int64),
            np.array(cur_r, dtype=np.float32))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vae = VAEEncoder(latent_dim=512).to(device).eval()
    vae.load_state_dict(torch.load("../world_model/checkpoints/vae_encoder.pth", map_location=device))

    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    obs_dec = ObsDecoder(400, 50, 512).to(device)
    reward_pred = RewardPredictor(400, 50).to(device)
    actor = Actor(400, 50, 2).to(device)
    critic = Critic(400, 50).to(device)

    # Load or create seed episodes
    data_path = "data/dreamer_trajectories.npz"
    data = np.load(data_path, allow_pickle=True)
    episodes = list(zip(data["o_seqs"], data["a_seqs"], data["r_seqs"]))
    print(f"Initial episodes: {len(episodes)}")

    N, C = 200, 100

    for it in range(N):
        wm_info = train_world_model(rssm, obs_dec, reward_pred,
                                     DreamerDataset(episodes), device, steps=C)
        o_seqs = [e[0] for e in episodes]; a_seqs = [e[1] for e in episodes]
        ac_info = train_ac(rssm, reward_pred, actor, critic,
                            o_seqs, a_seqs, device, steps=C)
        ep = collect_episode(vae, rssm, actor, device)
        episodes.append(ep)
        print(f"[{it+1:3d}/{N}] WM obs={wm_info['obs']:.4f} rew={wm_info['rew']:.4f} "
              f"kl={wm_info['kl']:.2f} | AC actor={ac_info['actor']:+.3f} "
              f"critic={ac_info['critic']:.3f} ret={ac_info['mean_ret']:+.2f} "
              f"ent={ac_info['entropy']:.3f} | collect {len(ep[0])-1} steps "
              f"r={ep[2].sum():.1f} buf={len(episodes)}")

    os.makedirs("checkpoints", exist_ok=True)
    for name, model in [("rssm", rssm), ("obs_decoder", obs_dec),
                         ("reward_predictor", reward_pred),
                         ("actor", actor), ("critic", critic)]:
        torch.save(model.state_dict(), f"checkpoints/{name}.pth")
    print("Saved.")


if __name__ == "__main__":
    main()
