import os
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from gymnasium.wrappers import AddRenderObservation

def collect_vae_data():
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)

    model = PPO.load("../CnnPolicy/flappy_bird_cnn_model")

    max_frames = 20000
    dataset = []

    print("Generating data...")
    obs, info = env.reset()

    while len(dataset) < max_frames:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = obs.transpose(2, 0, 1).astype(np.uint8)
        dataset.append(frame)
        
        if len(dataset) % 2000 == 0:
            print(f"Process: {len(dataset)} / {max_frames}")
            
        if terminated or truncated:
            obs, info = env.reset()

    env.close()

    dataset = np.array(dataset)
    np.savez_compressed("./data/flappy_bird_vae_dataset.npz", data=dataset)
    print("Saved to ./data/flappy_bird_vae_dataset.npz")

if __name__ == "__main__":
    collect_vae_data()