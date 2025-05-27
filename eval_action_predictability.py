#!/usr/bin/env python3
"""
Measure temporal-dependency score επ★ / επ for the Offline-DiffusionPolicy-UNet
checkpoint produced by train_rgbd.py (your pasted script).
"""

import argparse, json, os, random, numpy as np, torch, torch.nn as nn, torch.optim as optim
import gymnasium as gym
from diffusion_policy_unet_maniskill2 import Agent, make_env, SeqActionWrapper          # noqa: E402
from utils.ms2_data import load_demo_dataset                      # noqa: E402
from torch.utils.data import random_split, Dataset, DataLoader

# --------------------------------------------------------------------------- #
# 1. CLI
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt',      type=str, required=True,
                    help='checkpoint file saved by train_rgbd.py (e.g. .../checkpoints/best_eval_success_rate.pt)')
parser.add_argument('--demo-h5',   type=str, required=True, help='expert demo h5 (same dataset used in training)')
parser.add_argument('--device',    type=str, default='cuda')
parser.add_argument('--seed',      type=int, default=0)
parser.add_argument('--num-eval-envs', type=int, default=1)
parser.add_argument('--num-samples', type=int, default=1000, help='number of action samples to collect')

class ActionDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return x, y
        
        

def collect_policy_actions(env, agent, n_samples: int, args, device) -> np.ndarray:
    episodes = 0
    obs, _ = env.reset(seed=args.seed)
    num_envs = env.num_envs
    action_buffers = [None for _ in range(num_envs)]
    actions = []

    while len(actions) < n_samples:
        with torch.no_grad():
            act_seq = agent.get_eval_action(torch.tensor(obs, dtype=torch.float32, device=device))
        act_np = act_seq.cpu().numpy()

        obs, _r, term, trunc, info = env.step(act_np)
        for env_i in range(num_envs):
            if "executed_steps" in info:
                
                n_exec = info['executed_steps'][env_i]
                exec_act_seq = act_seq[:, :n_exec]
                if action_buffers[env_i] is not None:
                    prev_action = action_buffers[env_i]
                    actions.append((prev_action.reshape(-1).cpu().numpy(), exec_act_seq.reshape(-1).cpu().numpy()))
                    if len(actions) >= n_samples:
                        break
                action_buffers[env_i] = exec_act_seq

        done = np.logical_or(term, trunc)
        if done.any():
            # how many envs finished this step?
            n_finished = done.sum()
            episodes += n_finished.item()
            print(f"Episodes finished: {episodes}")
            # reset only the envs that finished, keep others running
            obs_reset, _ = env.reset(seed=args.seed+episodes)
            obs[done] = obs_reset
            for i, d in enumerate(done):
                if d:
                    action_buffers[i] = None
    env.close()
    return actions

def collect_expert_actions(h5_path:str) -> np.ndarray:
    traj = load_demo_dataset(h5_path, num_traj=None, concat=True)
    return traj['actions']

def collect_expert_chunk_pairs(data_path: str,
                               horizon: int,
                               n_samples: int = 50_000,
                               drop_last: bool = True):
    traj = load_demo_dataset(data_path, num_traj=None, concat=False)
    actions_list = traj['actions']        # list of arrays, each (T_i, act_dim)

    actions = []
    for acts in actions_list:
        T, act_dim = acts.shape
        # how many full chunks fit?
        n_chunks = T // horizon if drop_last else (T - horizon) // horizon
        for k in range(n_chunks - 1):     # need two consecutive chunks
            start_prev = k      * horizon
            start_curr = (k+1)  * horizon
            chunk_prev = acts[start_prev : start_prev + horizon]      # (H, act_dim)
            chunk_curr = acts[start_curr : start_curr + horizon]
            actions.append((chunk_prev.reshape(-1), chunk_curr.reshape(-1)))
            if len(actions) >= n_samples:
                return actions
    return actions

def mse_of_mlp(actions: list, device, epochs:int=40) -> float:
    dataset = ActionDataset(actions)
    # Shuffle / split
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    action_dim = actions[0][0].shape[-1]
    # breakpoint()
    dataloader = DataLoader(
        train_dataset,
        batch_size = 32,
        shuffle=True,
    )

    mlp = nn.Sequential(nn.Linear(action_dim, 64), nn.ReLU(),
                        nn.Linear(64, action_dim)).to(device)
    opt = optim.Adam(mlp.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            pred = mlp(xb)
            opt.zero_grad()
            loss_fn(pred, yb).backward()
            opt.step()
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        x_test, y_test = next(iter(test_loader))
        x_test, y_test = x_test.to(device), y_test.to(device)
        mse = loss_fn(mlp(x_test), y_test).item()
    return mse

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # --------------------------------------------------------------------------- #
    # Load checkpoint & its hyperparams
    # --------------------------------------------------------------------------- #
    ckpt = torch.load(args.ckpt, map_location=device)

    ckpt_dir  = os.path.dirname(args.ckpt)
    with open(os.path.join(os.path.dirname(ckpt_dir), 'args.json'), 'r') as f:
        train_args = argparse.Namespace(**json.load(f))

    # --------------------------------------------------------------------------- #
    # Rebuild env and policy
    # --------------------------------------------------------------------------- #
    VecEnv = gym.vector.SyncVectorEnv if args.num_eval_envs == 1 \
        else lambda x: gym.vector.AsyncVectorEnv(x, context='forkserver')
    env = VecEnv(
        [make_env(train_args.env_id, args.seed + 1000 + i, train_args.control_mode,
                f'eval_videos' if train_args.capture_video and i == 0 else None,
                other_kwargs=train_args.__dict__)
        for i in range(args.num_eval_envs)]
    )

    agent = Agent(env, train_args).to(device)
    agent.load_state_dict(ckpt['ema_agent'])   # use EMA weights
    agent.eval()
    
    print("Loading expert demos ...")
    expert_act = collect_expert_chunk_pairs(args.demo_h5, 4, args.num_samples)
    print(f"  found {len(expert_act)} expert actions")

    print("Sampling policy rollouts ...")
    policy_act = collect_policy_actions(env, agent, len(expert_act), args, device)
    print(f"  collected {len(policy_act)} executed actions")

    print("Training small MLP regressors ...")
    eps_pi   = mse_of_mlp(policy_act, device)
    eps_star = mse_of_mlp(expert_act, device)

    print("\n========== ACTION PREDICTABILITY ==========")
    print(f"ε_pi   (policy)  = {eps_pi:.6f}")
    print(f"ε_pi★  (expert)  = {eps_star:.6f}")
    print(f"ratio  (ε_star / ε_pi) = {eps_star / eps_pi:.3f}")
    print(" > 1 ⇒ policy copies past actions too much (copy-cat)")
    print(" < 1 ⇒ policy under-utilises temporal dependency")
