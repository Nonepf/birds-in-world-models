"""
Dreamer V1 Actor-Critic: REINFORCE in RSSM imagination.
Discrete actions require score-function estimator; analytic gradient only works
for continuous actions (TanhNormal).
"""
import numpy as np, torch, torch.nn.functional as F, torch.optim as optim
from models import RSSM, RewardPredictor, Actor, Critic


def sample_states(rssm, o_seqs, a_seqs, B, warmup, device):
    """Batch sample initial (h,z) from real trajectory positions."""
    oc, ac = [], []
    for _ in range(B):
        while True:
            idx = np.random.randint(0, len(o_seqs))
            if len(o_seqs[idx]) > warmup + 2: break
        s = np.random.randint(0, len(o_seqs[idx]) - warmup - 1)
        oc.append(o_seqs[idx][s:s+warmup].astype(np.float32))
        ac.append(a_seqs[idx][s:s+warmup].astype(np.int64))
    ob = torch.from_numpy(np.stack(oc)).to(device)
    ab = torch.from_numpy(np.stack(ac)).to(device)
    with torch.no_grad():
        hs, _, zp, *_ = rssm.forward_sequence(F.one_hot(ab, 2).float(), ob)
    return hs[:, -1, :], zp[:, -1, :]


def lambda_returns(rewards, values, gamma, lmbda):
    B, H = rewards.shape
    V = torch.zeros(B, H, device=rewards.device)
    run = values[:, -1]
    for t in range(H - 1, -1, -1):
        run = rewards[:, t] + gamma * ((1 - lmbda) * values[:, t + 1] + lmbda * run)
        V[:, t] = run
    return V


def train_ac(rssm, reward_pred, actor, critic, o_seqs, a_seqs, device, steps=100):
    opt_actor = optim.Adam(actor.parameters(), lr=8e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=8e-5)

    B, H, warmup = 128, 15, 5
    gamma, lmbda = 0.99, 0.95
    total_actor, total_critic, total_ret, total_ent = 0, 0, 0, 0

    for _ in range(steps):
        h, z = sample_states(rssm, o_seqs, a_seqs, B, warmup, device)

        log_probs, values, rewards, entropies = [], [], [], []

        for t in range(H):
            hd, zd = h.detach(), z.detach()

            logits = actor(hd, zd)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())

            values.append(critic(hd, zd))
            rewards.append(reward_pred(hd, zd).detach())

            a_oh = F.one_hot(a, 2).float()
            with torch.no_grad():
                h, zp, _ = rssm(z, a_oh, h); z = zp

        values.append(critic(h.detach(), z.detach()))

        R = torch.stack(rewards, dim=1).squeeze(-1)      # (B, H)
        V = torch.stack(values, dim=1).squeeze(-1)        # (B, H+1)
        lp = torch.stack(log_probs, dim=1)                 # (B, H)
        ent = torch.stack(entropies, dim=1).mean()         # scalar

        V_lambda = lambda_returns(R, V, gamma, lmbda)
        advantage = (V_lambda - V[:, :-1]).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        actor_loss = -(lp * advantage).mean() - 0.1 * ent
        critic_loss = F.mse_loss(V[:, :-1], V_lambda.detach())

        opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()
        opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()

        total_actor += actor_loss.item(); total_critic += critic_loss.item()
        total_ret += V_lambda.mean().item(); total_ent += ent.item()

    n = max(steps, 1)
    return {"actor": total_actor/n, "critic": total_critic/n,
            "mean_ret": total_ret/n, "entropy": total_ent/n}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    rssm.load_state_dict(torch.load("checkpoints/rssm.pth", map_location=device))
    reward_pred = RewardPredictor(400, 50).to(device)
    reward_pred.load_state_dict(torch.load("checkpoints/reward_predictor.pth", map_location=device))

    actor = Actor(400, 50, 2).to(device)
    critic = Critic(400, 50).to(device)

    data = np.load("data/dreamer_trajectories.npz", allow_pickle=True)
    o_seqs, a_seqs = data["o_seqs"], data["a_seqs"]

    for i in range(200):
        info = train_ac(rssm, reward_pred, actor, critic, o_seqs, a_seqs, device, steps=100)
        print(f"Iter [{i+1:3d}/200] actor={info['actor']:+.4f}  "
              f"critic={info['critic']:.4f}  ret={info['mean_ret']:+.3f}  "
              f"ent={info['entropy']:.3f}")

    import os; os.makedirs("checkpoints", exist_ok=True)
    torch.save(actor.state_dict(), "checkpoints/actor.pth")
    torch.save(critic.state_dict(), "checkpoints/critic.pth")
    print("Saved.")


if __name__ == "__main__":
    main()
