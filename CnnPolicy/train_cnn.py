import gymnasium as gym
import flappy_bird_gymnasium
from stable_baselines3 import PPO

env = gym.make("FlappyBird-rgb-v0", render_mode="human")

model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        learning_rate=1e-4
        tensorboard_log="./tb_logs/"
)

model.learn(total_timesteps=100000, tg_log_name="flappy_bird_rnnpolicy")

model.save("flappy_bird_rnnpolicy")
env.close()
