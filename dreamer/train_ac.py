"""
Train Actor-Critic in RSSM imagination.
"""
import torch, torch.nn.functional as F, torch.optim as optim

from models import RSSM
from models import RewardPredictor, Actor, Critic


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    rssm.load_state_dict(torch.load("checkpoints/rssm.pth", map_location=device))
    rssm.eval()

    reward_pred = RewardPredictor(400, 50).to(device)
    reward_pred.load_state_dict(torch.load("checkpoints/reward_predictor.pth", map_location=device))
    reward_pred.eval()

    actor = Actor(400, 50, 2).to(device)
    critic = Critic(400, 50).to(device)
    opt_actor = optim.Adam(actor.parameters(), lr=1e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=1e-4)

    horizon = 15
    steps = 5000

    for step in range(steps):
        B = 16
        h = torch.zeros(B, 400, device=device)
        z = torch.zeros(B, 50, device=device)

        log_probs_all, values_all, rewards_all = [], [], []

        for _ in range(horizon):
            action_logits = actor(h, z)
            dist = torch.distributions.Categorical(logits=action_logits)
            a = dist.sample()
            log_prob = dist.log_prob(a)
            value = critic(h, z)

            a_onehot = F.one_hot(a, num_classes=2).float()

            with torch.no_grad():
                h, z_prior, _ = rssm(z, a_onehot, h)
                z = z_prior
                r_pred = reward_pred(h, z)

            log_probs_all.append(log_prob)
            values_all.append(value.squeeze(-1))
            rewards_all.append(r_pred.squeeze(-1))

        returns = torch.stack(rewards_all, dim=1)
        gamma = 0.99
        disc_returns = torch.zeros_like(returns)
        running = torch.zeros(B, device=device)
        for t in range(horizon - 1, -1, -1):
            running = returns[:, t] + gamma * running
            disc_returns[:, t] = running

        values = torch.stack(values_all, dim=1)
        log_probs = torch.stack(log_probs_all, dim=1)
        advantage = disc_returns - values.detach()

        actor_loss = -(log_probs * advantage).mean()
        critic_loss = F.mse_loss(values, disc_returns)

        opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()
        opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()

        if (step + 1) % 500 == 0:
            print(f"Step [{step+1:4d}/{steps}] actor={actor_loss.item():.4f}  "
                  f"critic={critic_loss.item():.4f}  mean_ret={disc_returns.mean().item():.2f}")

    torch.save(actor.state_dict(), "checkpoints/actor.pth")
    torch.save(critic.state_dict(), "checkpoints/critic.pth")
    print("Saved to checkpoints/")


if __name__ == "__main__":
    main()
