import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from gymnasium.wrappers import AddRenderObservation, HumanRendering

env = gym.make("FlappyBird-v0", render_mode="rgb_array")
env = AddRenderObservation(env, render_only=True)
env = HumanRendering(env)

model = PPO.load("flappy_bird_cnn_model")

print("Evaluation Started.")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
    if terminated or truncated:
        print(f"Game Over. Score: {info.get('score', 0)}")
        obs, info = env.reset()

# env.close()