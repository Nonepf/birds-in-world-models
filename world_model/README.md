# World Model

This folder applies [world model 2018](https://arxiv.org/abs/1803.10122) to the Flappy Bird game. Our implementation is informed by [world-models](https://github.com/ctallec/world-models), a PyTorch reproduction of the original paper.

## Architecture Overview

The following graph demonstrate the main architecture of the model.

```mermaid
graph TD
    classDef default fill:#ffffff,stroke:#333,stroke-width:1px,color:#000000;

    ENV["Gymnasium Environment"]:::default
    V["V (VAE)"]:::default
    C["C (Linear)"]:::default
    M["M (MDN-RNN)"]:::default

    ENV -->|Observation| V
    V -->|Code z| C
    V -->|Code z| M
    M -->|Hidden state h| C
    C -->|Action a| ENV
```



