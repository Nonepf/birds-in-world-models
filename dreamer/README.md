# Dreamer

PyTorch reproduction of [DreamerV1](https://arxiv.org/abs/1912.01603), applied to Flappy Bird. Reuses the VAE encoder from [world_model](../world_model/) for observation compression.

## Pipeline

```bash
cd dreamer
python generate_dreamer_data.py   # Step 1: collect trajectories (CNN + random)
python train_world_model.py       # Step 2: train RSSM world model
python train_ac.py                # Step 3: train Actor (CMA-ES on real env)
python demo.py                    # Play
```

Requires a trained VAE checkpoint at `../world_model/checkpoints/vae_encoder.pth`.

## Design notes

### Data mixing

CNN policy alone creates a spurious correlation: flapping co-occurs with dying because the CNN only flaps when already in danger. A reward predictor trained on this data learns that flap → low reward, which kills imagination-based RL. Mixing random-action episodes (50/50) breaks this correlation.

### Actor training

We use CMA-ES directly on the real environment (not imagination). The Actor's first layer is pretrained via behavioral cloning, then frozen. Only the output layer (802 parameters) is optimized by CMA-ES against real game rewards. This avoids the distribution-shift and reward-predictor-quality issues that plague pure imagination training.

### GPU utilization

- RSSM uses `nn.GRU` (not `nn.GRUCell`) for optimized CUDA sequence kernels
- World model training batches all MLP head predictions over `(B*T)` in a single forward pass
- DataLoader uses `num_workers=4` with `pin_memory=True`
