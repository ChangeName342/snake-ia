import torch
import numpy as np
from env.snake_env import DIRECTIONS, DIR_TO_IDX
from agents.astar_agent import expert_agent, astar_path, path_to_direction

import torch.nn as nn
import torch.optim as optim
import random

class PolicyNet(nn.Module):
    def __init__(self, board_size=10):
        super().__init__()
        in_ch = 2
        flat = in_ch * board_size * board_size
        self.net = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, -1)
        return self.net(x)

def obs_to_tensor(obs):
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

def collect_expert_data(env, max_episodes=200, max_steps_per_episode=500):
    dataset = []
    for ep in range(max_episodes):
        obs = env.reset()
        for step in range(max_steps_per_episode):
            path = astar_path(env)
            if path and len(path)>=2:
                action = path_to_direction(path)
            else:
                acts = env.available_actions()
                action = random.choice(acts) if acts else None
            if action is None:
                break
            dataset.append((obs.copy(), DIR_TO_IDX[action]))
            obs, r, done, info = env.step(action)
            if done:
                break
        if len(dataset) > 2000:
            break
    return dataset

def train_imitation(model, dataset, epochs=10, batch_size=64, lr=1e-3):
    if not dataset:
        return
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    X = np.array([d[0] for d in dataset])
    Y = np.array([d[1] for d in dataset])
    n = len(X)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.tensor(X[idx], dtype=torch.float32).to(device)
            yb = torch.tensor(Y[idx], dtype=torch.long).to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.to('cpu')

def model_agent_factory(model):
    def agent(obs, env):
        tensor = obs_to_tensor(obs)
        with torch.no_grad():
            logits = model(tensor).squeeze(0).numpy()
        valid = {}
        for d in DIRECTIONS:
            valid[DIR_TO_IDX[d]] = True
        rev = (-env.direction.dx, -env.direction.dy)
        for d in DIRECTIONS:
            if (d.dx, d.dy) == rev:
                valid[DIR_TO_IDX[d]] = False
        for d in DIRECTIONS:
            idx = DIR_TO_IDX[d]
            if not valid[idx]:
                continue
            hx, hy = env.snake[0]
            nx, ny = hx+d.dx, hy+d.dy
            if not (0<=nx<env.size and 0<=ny<env.size):
                valid[idx]=False
            elif (nx,ny) in set(env.snake) and (nx,ny)!=env.snake[-1]:
                valid[idx]=False
        best_idx=None
        best_val=-1e9
        for idx, ok in valid.items():
            if not ok:
                continue
            if logits[idx]>best_val:
                best_val=logits[idx]
                best_idx=idx
        if best_idx is None:
            acts = env.available_actions()
            return random.choice(acts) if acts else None
        for d in DIRECTIONS:
            if DIR_TO_IDX[d]==best_idx:
                return d
        return None
    return agent