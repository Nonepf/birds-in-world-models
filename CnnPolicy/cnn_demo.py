import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO

env = gym.make("FlappyBird-v0", render_mode="human")
model = PPO.load("flappy_bird_cnnpolicy")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
