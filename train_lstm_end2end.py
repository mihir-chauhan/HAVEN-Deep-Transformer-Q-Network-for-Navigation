import os
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sim_core import (
    SimulationConfig,
    LSTMEnd2EndActionPolicy,
    LSTMPointRegressor,
    Agent,
    generate_enemies,
    generate_obstacles,
    add_goal_as_obstacle,
    obstacle_force,
    desired_force,
    anticipatory_enemy_avoidance_force,
)


def teacher_action(agent, enemies, obstacles_shapely, cfg: SimulationConfig) -> np.ndarray:
    # Potential-field style teacher (same components used in baselines)
    F_desired = desired_force(agent, desired_speed=cfg.desired_speed) * 10.0
    F_obs = obstacle_force(agent, obstacles_shapely, threshold=cfg.obstacle_threshold, repulsive_coeff=cfg.obstacle_repulsive)
    F_anticipatory = np.zeros(2)
    for e in enemies:
        F_anticipatory += anticipatory_enemy_avoidance_force(agent, e, T=cfg.exposure_T_pred, weight=cfg.anticipate_weight)
    F_total = F_desired + F_obs + F_anticipatory
    # Clamp to a reasonable magnitude (matches action proposer max_mag ~ 2.0)
    mag = np.linalg.norm(F_total)
    if mag > 2.0:
        F_total = (F_total / mag) * 2.0
    return F_total


def collect_bc_dataset(num_envs: int, steps_per_env: int, cfg: SimulationConfig, rng_seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(rng_seed)
    feats: List[np.ndarray] = []
    acts: List[np.ndarray] = []

    # Use the same feature encoder as the policy to match input_dim ordering
    encoder = LSTMEnd2EndActionPolicy(k=3, checkpoint_path=None)

    for env_idx in range(num_envs):
        agent_start = np.array([cfg.scene_min, cfg.scene_min], dtype=float)
        agent_goal = np.array([cfg.scene_max, cfg.scene_max], dtype=float)
        agent = Agent(agent_start, agent_goal, max_speed=1.0)
        enemies = generate_enemies(cfg, np.random.default_rng(rng_seed + env_idx))
        obstacles, obstacles_shapely = generate_obstacles(cfg, np.random.default_rng(rng_seed + env_idx + 42))
        add_goal_as_obstacle(obstacles, obstacles_shapely, agent_goal)

        for _ in range(steps_per_env):
            # Encode current state
            v = encoder.encode_state(agent, agent_goal, enemies, obstacles_shapely)
            # Teacher action
            a = teacher_action(agent, enemies, obstacles_shapely, cfg)
            feats.append(v.astype(np.float32))
            acts.append(a.astype(np.float32))
            # Move agent a bit using teacher to diversify
            agent.velocity = 0.9 * agent.velocity + 0.1 * a
            sp = np.linalg.norm(agent.velocity)
            if sp > 2.0:
                agent.velocity = (agent.velocity / sp) * 2.0
            agent.pos = agent.pos + agent.velocity * cfg.dt

    X = np.stack(feats, axis=0) if feats else np.zeros((0, 12), dtype=np.float32)
    Y = np.stack(acts, axis=0) if acts else np.zeros((0, 2), dtype=np.float32)
    return X, Y


def train_lstm_e2e_action(X: np.ndarray, Y: np.ndarray, k: int, epochs: int, batch_size: int, lr: float, device: str) -> LSTMPointRegressor:
    input_dim = X.shape[1]
    model = LSTMPointRegressor(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            xseq = torch.tile(xb.unsqueeze(1), (1, k, 1))
            out_seq = model(xseq)  # (B, k, 2)
            out_last = out_seq[:, -1, :]
            loss = loss_fn(out_last, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def main():
    parser = argparse.ArgumentParser(description='Train LSTM end-to-end action via behavior cloning from PF teacher.')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--steps-per-episode', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--out', type=str, default='lstm_end2end_checkpoint.pt')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = SimulationConfig()

    X, Y = collect_bc_dataset(
        num_envs=args.episodes,
        steps_per_env=args.steps_per_episode,
        cfg=cfg,
        rng_seed=args.seed,
    )

    if X.shape[0] == 0:
        print('[train_lstm_end2end] No data collected. Aborting.')
        return

    model = train_lstm_e2e_action(
        X=X,
        Y=Y,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    torch.save(model.state_dict(), out_path)
    print(f'[train_lstm_end2end] Saved checkpoint to {out_path}')


if __name__ == '__main__':
    main()


