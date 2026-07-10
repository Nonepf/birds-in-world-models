# Dreamer

This folder is a PyTorch reproduction of the [dreamer paper](https://arxiv.org/abs/1912.01603). The project is informed by [Dreamer](https://github.com/google-research/dreamer), a tensorflow implementation of the dreamer. We adopted their design philosophy and implemented it on PyTorch.

Note: this implementation reused the VAE model in [world_model](../world_model/README.md). We freezed the parameter and use it for dimensionality reduction.

With limited time, we lean heavily on AI coding agents to handle much of our coding.

## Quick Start 

```bash
cd dreamer
python generate_dreamer_data.py
python train_world_model.py
python train_ac.py
```

Make sure a VAE `.pth` model is available in `world_model/checkpoints`.

## Supplement

The original version performs poorly, consistently receiving zero score. 

<table>
  <tr>
    <td><img src="images/demo1_old.gif" width="300"></td>
    <td><img src="images/demo2_old.gif" width="300"></td>
    <td><img src="images/demo3_old.gif" width="300"></td>
  </tr>
</table>

The root cause of this issue may lie in a training–inference mismatch. During training, the model's reasoning process is iteratively conditioned on its own output distribution at each step, mimicking an autoregressive rollout. However, this deviates from the actual deployment conditions, leading to cumulative distributional drift that ultimately causes the model to collapse. 

This phenomenon closely resembles the exposure bias problem in LLM training, and the mitigation strategies can be similarly adapted from established LLM practices.

A demonstration of the improved version is provided below.



---
