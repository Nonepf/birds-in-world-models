"""
Train Actor via behavioral cloning from CNN policy trajectories.
"""
import numpy as np
import torch, torch.nn.functional as F, torch.optim as optim

from models import RSSM, Actor


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    rssm = RSSM(z_dim=50, h_dim=400, action_dim=2, obs_dim=512).to(device)
    rssm.load_state_dict(torch.load("checkpoints/rssm.pth", map_location=device))
    rssm.eval()

    data = np.load("data/dreamer_trajectories.npz", allow_pickle=True)
    o_seqs = data["o_seqs"]
    a_seqs = data["a_seqs"]

    actor = Actor(400, 50, 2).to(device)
    opt = optim.Adam(actor.parameters(), lr=1e-3)

    epochs = 30
    batch_size = 64
    seg_len = 20

    for epoch in range(epochs):
        actor.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for _ in range(200):
            o_batch, a_batch = [], []
            for _ in range(batch_size):
                while True:
                    idx = np.random.randint(0, len(o_seqs))
                    if len(o_seqs[idx]) > seg_len + 2:
                        break
                T = len(o_seqs[idx]) - 2
                start = np.random.randint(0, T - seg_len)
                o_batch.append(o_seqs[idx][start:start + seg_len].astype(np.float32))
                a_batch.append(a_seqs[idx][start:start + seg_len].astype(np.int64))

            o_tensor = torch.from_numpy(np.stack(o_batch)).to(device)  # (B, seg_len, 512)
            a_tensor = torch.from_numpy(np.stack(a_batch)).to(device)  # (B, seg_len)

            B, L, _ = o_tensor.shape

            opt.zero_grad()

            h = torch.zeros(B, 400, device=device)
            z = torch.zeros(B, 50, device=device)

            loss_sum = 0
            correct = 0
            total = 0

            for t in range(L - 1):
                o_t = o_tensor[:, t, :]        # (B, 512)
                a_t = a_tensor[:, t]           # (B,)

                a_onehot = F.one_hot(a_t, num_classes=2).float()
                h, _, z = rssm(z, a_onehot, h, o_t)

                logits = actor(h, z)           # (B, 2)
                target = a_tensor[:, t + 1]    # (B,)

                loss_sum += F.cross_entropy(logits, target)
                correct += (torch.argmax(logits, dim=1) == target).sum().item()
                total += B

            loss = loss_sum / (L - 1)
            loss.backward() #type: ignore
            opt.step()

            total_loss += loss.item() * total #type: ignore
            total_correct += correct
            total_samples += total

        acc = total_correct / max(total_samples, 1)
        print(f"Epoch [{epoch+1:2d}/{epochs}] loss={total_loss/max(total_samples,1):.4f}  acc={acc:.3f}")

        # Check flap rate on a test batch
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                o_test, _ = [], []
                for _ in range(100):
                    idx = np.random.randint(0, len(o_seqs))
                    o_test.append(o_seqs[idx][:seg_len].astype(np.float32))
                o_test = torch.from_numpy(np.stack(o_test)).to(device)
                B = o_test.shape[0]
                L = o_test.shape[1]
                h = torch.zeros(B, 400, device=device)
                z = torch.zeros(B, 50, device=device)
                dummy_a = torch.zeros(B, 2, device=device)
                actions = []
                for t in range(L - 1):
                    h, _, z = rssm(z, dummy_a, h, o_test[:, t, :])
                    a = torch.argmax(actor(h, z), dim=1)
                    actions.append(a.cpu().numpy())
                    dummy_a = F.one_hot(a, num_classes=2).float()
                all_a = np.concatenate(actions)
                flap = all_a.mean()
                print(f"  -> flap_rate={flap:.2f}")

    torch.save(actor.state_dict(), "checkpoints/actor.pth")
    print("Saved to checkpoints/")


if __name__ == "__main__":
    main()
