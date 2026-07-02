# birds-in-world-models
Reproducing the evolution of World Models (from Ha et al. (2018) to Dreamer, Genie, etc.) from scratch in PyTorch using Flappy Bird.

## Introduction

This repository contains clean, from-scratch PyTorch implementations of various famous World Models. We rewrite the core code, train and evaluate these models on a gym-compatible flappy bird environment.

### Roadmap

We plan to implement and evaluate the following architectures:

- [ ] World Model (2018)
- [ ] Dreamer V1
- [ ] Genie
- [ ] V-JEPA-AC 

### Current Progress

- [x] Initialize repository setup.
- [ ] Implement a simple CNN to test the environment.
- [ ] Prepare data for training the world model(2018).

## Installation and Quick Start

```bash
git clone https://github.com/Nonepf/birds-in-world-models.git
pip install requirements.txt
pip install flappy-bird-gymnasium
```

View [CnnPolicy](CnnPolicy/README.md) to have a glance at a simple CNN implementation.

## Liscence

This project is licensed under the MIT License.