"""
Train Actor-Critic in RSSM imagination (Dreamer V1).
Phase 1: BC pretrain (behavioral cloning from data) to avoid initial collapse.
Phase 2: AC training with entropy bonus.
"""
import numpy as np
import torch, torch.nn.functional as F, torch.optim as optim

from models import RSSM, RewardPredictor, Actor, Critic


def sample_initial(rssm, o_seqs, a_seqs, B, warmup, device):
    o_chunks, a_chunks = [], []
    for _ in range(B):
        while True:
            idx = np.random.randint(0, len(o_seqs))
            if len(o_seqs[idx]) > warmup + 2:
                break
        start = np.random.randint(0, len(o_seqs[idx]) - warmup - 1)
        o_chunks.append(o_seqs[idx][start:start + warmup].astype(np.float32))
        a_chunks.append(a_seqs[idx][start:start + warmup].astype(np.int64))

    o_b = torch.from_numpy(np.stack(o_chunks)).to(device)
    a_b = torch.from_numpy(np.stack(a_chunks)).to(device)
    a_oh = F.one_hot(a_b, 2).float()

    with torch.no_grad():
        h_seq, _, z_post = rssm.forward_sequence(a_oh, o_b)
    return h_seq[:, -1, :], z_post[:, -1, :]


def lambda_returns(rewards, values, gamma, lmbda):
    B, H = rewards.shape
    V = torch.zeros(B, H, device=rewards.device)
    running = values[:, -1]
    for t in range(H - 1, -1, -1):
        running = rewards[:, t] + gamma * ((1 - lmbda) * values[:, t + 1] + lmbda * running)
        V[:, t] = running
    return V


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    rssm.load_state_dict(torch.load("checkpoints/rssm.pth", map_location=device))

    reward_pred = RewardPredictor(400, 50).to(device)
    reward_pred.load_state_dict(torch.load("checkpoints/reward_predictor.pth", map_location=device))

    data = np.load("data/dreamer_trajectories.npz", allow_pickle=True)
    o_seqs, a_seqs = data["o_seqs"], data["a_seqs"]

    actor = Actor(400, 50, 2).to(device)
    critic = Critic(400, 50).to(device)

    # === Phase 1: BC pretrain ===
    print("Phase 1: BC pretraining...")
    rssm.train()  # cuDNN backward needs train mode
    opt_bc = optim.Adam(actor.parameters(), lr=1e-3)
    B_bc, L = 128, 30

    for epoch in range(8):
        total_loss, total_correct, total = 0, 0, 0
        for _ in range(80):
            o_b, a_b = [], []
            for _b in range(B_bc):
                while True:
                    idx = np.random.randint(0, len(o_seqs))
                    if len(o_seqs[idx]) > L + 2:
                        break
                start = np.random.randint(0, len(o_seqs[idx]) - L - 1)
                o_b.append(o_seqs[idx][start:start + L].astype(np.float32))
                a_b.append(a_seqs[idx][start:start + L].astype(np.int64))
            o_t = torch.from_numpy(np.stack(o_b)).to(device)
            a_t = torch.from_numpy(np.stack(a_b)).to(device)

            opt_bc.zero_grad()
            a_oh = F.one_hot(a_t, 2).float()
            h_seq, _, z_post = rssm.forward_sequence(a_oh, o_t)
            h_in = h_seq[:, :-1, :].reshape(B_bc * (L - 1), 400)
            z_in = z_post[:, :-1, :].reshape(B_bc * (L - 1), 50)
            logits = actor(h_in, z_in)
            targets = a_t[:, 1:].reshape(B_bc * (L - 1))
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt_bc.step()
            correct = (torch.argmax(logits, dim=1) == targets).sum().item()
            total_loss += loss.item() * B_bc * (L - 1)
            total_correct += correct
            total += B_bc * (L - 1)
        print(f"  BC [{epoch+1}/8] loss={total_loss/max(total,1):.4f} acc={total_correct/max(total,1):.3f}")

    torch.save(actor.state_dict(), "checkpoints/actor.pth")
    print("BC pretrain done.")

    # === Phase 2: AC in imagination ===
    print("\nPhase 2: Actor-Critic in imagination...")
    rssm.eval()
    reward_pred.eval()

    opt_actor = optim.Adam(actor.parameters(), lr=8e-5)
    opt_critic = optim.Adam(critic.parameters(), lr=8e-5)

    B, H, warmup = 64, 15, 5
    gamma, lmbda = 0.99, 0.95
    steps = 10000

    for step in range(steps):
        h, z = sample_initial(rssm, o_seqs, a_seqs, B, warmup, device)

        log_probs, values_all, rewards_all, entropies = [], [], [], []

        for t in range(H):
            logits = actor(h, z)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            log_prob = dist.log_prob(a)
            entropy = dist.entropy()

            v = critic(h, z)
            values_all.append(v.squeeze(-1))
            rewards_all.append(reward_pred(h, z).squeeze(-1))
            log_probs.append(log_prob)
            entropies.append(entropy)

            a_oh = F.one_hot(a, 2).float()
            with torch.no_grad():
                h, z_prior, _ = rssm(z, a_oh, h)
                z = z_prior

        with torch.no_grad():
            v_final = critic(h, z).squeeze(-1)
        values_all.append(v_final)

        rewards = torch.stack(rewards_all, dim=1)     # (B, H)
        values = torch.stack(values_all, dim=1)        # (B, H+1)
        log_probs = torch.stack(log_probs, dim=1)

        V_lambda = lambda_returns(rewards, values, gamma, lmbda)

        advantage = V_lambda - values[:, :-1].detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        entropy = torch.stack(entropies, dim=1).mean()
        ent_coef = max(0.01, 0.1 * (1 - step / 5000))

        actor_loss = -(log_probs * advantage.detach()).mean() - ent_coef * entropy
        critic_loss = F.mse_loss(values[:, :-1], V_lambda.detach())

        opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()
        opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()

        if step == 0:
            print(f"  [debug] r_mean={rewards.mean().item():.4f} r_std={rewards.std().item():.4f}")
            print(f"  [debug] v_mean={values.mean().item():.4f} v_std={values.std().item():.4f} v_bootstrap={values[:,-1].mean().item():.4f}")
            print(f"  [debug] V_lambda mean={V_lambda.mean().item():.4f} std={V_lambda.std().item():.4f}")

        if (step + 1) % 500 == 0:
            with torch.no_grad():
                h_t, z_t = sample_initial(rssm, o_seqs, a_seqs, 128, warmup, device)
                test_a = torch.argmax(actor(h_t, z_t), dim=1)
                flap = test_a.float().mean().item()
            print(f"Step [{step+1:5d}/{steps}] actor={actor_loss.item():.4f}  "
                  f"critic={critic_loss.item():.4f}  mean_ret={V_lambda.mean().item():.3f}  "
                  f"flap={flap:.2f}  ent={entropy.item():.3f}")

    torch.save(actor.state_dict(), "checkpoints/actor.pth")
    torch.save(critic.state_dict(), "checkpoints/critic.pth")
    print("Saved to checkpoints/")


if __name__ == "__main__":
    main()
