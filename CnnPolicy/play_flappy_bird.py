import gymnasium as gym
import flappy_bird_gymnasium
import pygame
import sys

env = gym.make("FlappyBird-v0", render_mode="human")
obs, info = env.reset()

print("Game Start!")

clock = pygame.time.Clock()

while True:
    action = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1
            elif event.key == pygame.K_ESCAPE:
                env.close()
                sys.exit()

    obs, reward, terminated, truncated, info = env.step(action)
    # print(obs)

    if terminated or truncated:
        print(f"Game Over!\nScore: {info.get('score', 0)}")
        obs, info = env.reset()

    clock.tick(30)
