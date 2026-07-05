import os
import torch
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium.wrappers import AddRenderObservation, HumanRendering

from models import Encoder, MDNRNN, Controller

def watch_world_model_play():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    encoder = Encoder(latent_dim=512).to(device)
    encoder.load_state_dict(torch.load("./checkpoints/vae_encoder.pth", map_location=device))
    encoder.eval()

    rnn = MDNRNN(z_dim=512, hidden_dim=256).to(device)
    rnn.load_state_dict(torch.load("./checkpoints/mdn_rnn_world_model.pth", map_location=device))
    rnn.eval()

    controller = Controller(z_dim=512, hidden_dim=256, action_dim=2).to(device)
    controller.load_state_dict(torch.load("./checkpoints/controller_best.pth", map_location=device))
    controller.eval()

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)
    env = HumanRendering(env)
    
    obs, info = env.reset()
    
    hidden = (torch.zeros(1, 1, rnn.hidden_dim).to(device),
              torch.zeros(1, 1, rnn.hidden_dim).to(device))

    with torch.no_grad():
        while True:
            frame = obs.transpose(2, 0, 1).astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)
            z, _ = encoder(frame_tensor)
            
            h = hidden[0].squeeze(0)
            
            action_logits = controller(z, h)
            action = torch.argmax(action_logits, dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            act_tensor = torch.tensor([[action]], device=device)
            act_embed = rnn.action_embed(act_tensor)  # (1, 1, 16)
            lstm_input = torch.cat([z.unsqueeze(1), act_embed], dim=-1)
            _, hidden = rnn.lstm(lstm_input, hidden)

            if terminated or truncated:
                print(f"Score: {info.get('score', 0)}")
                
                obs, info = env.reset()
                hidden = (torch.zeros(1, 1, rnn.hidden_dim).to(device),
                          torch.zeros(1, 1, rnn.hidden_dim).to(device))

if __name__ == "__main__":
    watch_world_model_play()