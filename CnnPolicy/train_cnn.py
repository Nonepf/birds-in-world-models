import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from gymnasium.wrappers import AddRenderObservation

env = gym.make("FlappyBird-v0", render_mode="rgb_array")

env = AddRenderObservation(env, render_only=True)

print(f"Observation Space: {env.observation_space}")

model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-4,
        tensorboard_log="./tb_logs/",
        policy_kwargs={
            "features_extractor_kwargs": {"features_dim": 256},
            "normalize_images": True,
        }
)

model.learn(total_timesteps=100000, tb_log_name="flappy_bird_cnn_model")
model.save("flappy_bird_cnn_model")
env.close()