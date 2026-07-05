import os
import torch
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium
from gymnasium.wrappers import AddRenderObservation
import cma

from models import Encoder, MDNRNN, Controller

def evaluate_params(params, encoder, rnn, controller, env, device):
    with torch.no_grad():
        state_dict = controller.state_dict()
        current_idx = 0
        for name, param in state_dict.items():
            num_elements = param.numel()
            flat_slice = params[current_idx : current_idx + num_elements]
            param.copy_(torch.tensor(flat_slice, dtype=torch.float32).view_as(param))
            current_idx += num_elements

    obs, info = env.reset()
    hidden = (torch.zeros(1, 1, rnn.hidden_dim).to(device),
              torch.zeros(1, 1, rnn.hidden_dim).to(device))
    total_reward = 0
    terminated, truncated = False, False
    
    with torch.no_grad():
        while not (terminated or truncated):
            frame = obs.transpose(2, 0, 1).astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame).unsqueeze(0).to(device)
            z, _ = encoder(frame_tensor)
            
            h = hidden[0].squeeze(0)
            action_logits = controller(z, h)
            action = torch.argmax(action_logits, dim=1).item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            act_float = torch.tensor([[[float(action)]]], device=device)
            lstm_input = torch.cat([z.unsqueeze(1), act_float], dim=-1)
            _, hidden = rnn.lstm(lstm_input, hidden)
            
    return total_reward

def train_with_cma():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder = Encoder(latent_dim=512).to(device)
    encoder.load_state_dict(torch.load("./checkpoints/vae_encoder.pth", map_location=device))
    encoder.eval()
    
    rnn = MDNRNN(z_dim=512, hidden_dim=256).to(device)
    rnn.load_state_dict(torch.load("./checkpoints/mdn_rnn_world_model.pth", map_location=device))
    rnn.eval()
    
    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = AddRenderObservation(env, render_only=True)
    
    base_controller = Controller(z_dim=512, hidden_dim=256, action_dim=2).to(device)
    
    total_params_num = sum(p.numel() for p in base_controller.parameters())
    print(f"total_params_num: {total_params_num}")
    
    es = cma.CMAEvolutionStrategy(total_params_num * [0.0], 0.2)
    
    generation = 0
    while not es.stop() and generation < 50:
        generation += 1
        solutions = es.ask() 
        scores = []
        
        for params in solutions:
            score = evaluate_params(params, encoder, rnn, base_controller, env, device)
            scores.append(score)
            
        costs = [-score for score in scores]
        es.tell(solutions, costs)
        
        print(f"Generation {generation} | Max Score: {max(scores):.1f} | Avg Score: {np.mean(scores):.1f}")
        
        if generation % 5 == 0:
            best_params = es.result.xbest
            evaluate_params(best_params, encoder, rnn, base_controller, env, device)
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save(base_controller.state_dict(), "./checkpoints/controller_best.pth")

    env.close()
    print("Saved to ./checkpoints/controller_best.pth")

if __name__ == "__main__":
    train_with_cma()