import os
import argparse
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sim_core import (
    SimulationConfig,
    DTQNHighLevelPolicy,
    LSTMHighLevelPolicy,
    LSTMQRegressor,
    Agent,
    generate_enemies,
    generate_obstacles,
    add_goal_as_obstacle,
    build_candidate_obstacles,
    compute_action_mask,
)


def collect_dataset(
    num_envs: int,
    steps_per_env: int,
    cfg: SimulationConfig,
    teacher_dtqn_ckpt: Optional[str],
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect (features, target_q) pairs by sampling environments and scoring each valid candidate.
    If a DTQN checkpoint is provided, distill its Q-values. Otherwise, fall back to a heuristic score.
    """
    np.random.seed(rng_seed)

    encoder = LSTMHighLevelPolicy(k=3, eps_greedy=0.0, checkpoint_path=None)

    teacher: Optional[DTQNHighLevelPolicy] = None
    if teacher_dtqn_ckpt and os.path.isfile(teacher_dtqn_ckpt):
        teacher = DTQNHighLevelPolicy(k=3, eps_greedy=0.0, checkpoint_path=teacher_dtqn_ckpt)

    feats: List[np.ndarray] = []
    targets: List[float] = []

    for env_idx in range(num_envs):
        # Build environment
        agent_start = np.array([cfg.scene_min, cfg.scene_min], dtype=float)
        agent_goal = np.array([cfg.scene_max, cfg.scene_max], dtype=float)
        agent = Agent(agent_start, agent_goal, max_speed=1.0)
        enemies = generate_enemies(cfg, np.random.default_rng(rng_seed + env_idx))
        obstacles, obstacles_shapely = generate_obstacles(cfg, np.random.default_rng(rng_seed + env_idx + 1337))
        add_goal_as_obstacle(obstacles, obstacles_shapely, agent_goal)

        for _ in range(steps_per_env):
            candidates = build_candidate_obstacles(obstacles)
            mask = compute_action_mask(candidates, agent, agent_goal)
            if not np.any(mask):
                # Move agent a little toward goal to vary state
                agent.pos = agent.pos + (agent_goal - agent.pos) * 0.05
                continue

            valid_indices = [i for i, m in enumerate(mask) if m]
            for i in valid_indices:
                v = encoder.encode_state_candidate(agent, enemies, candidates[i], agent_goal, obstacles_shapely)
                # Teacher target Q
                if teacher is not None:
                    # Prepare DTQN input and forward to get Q
                    teacher._ensure_model(len(v))
                    with torch.no_grad():
                        x = torch.from_numpy(np.tile(v, (teacher.k, 1))).float().unsqueeze(0)
                        q = teacher.model(x)[:, -1, 0].item()
                    y = float(q)
                else:
                    # Heuristic: prefer candidates closer to goal, farther from enemies, fewer seeing enemies
                    agent_to_c = candidates[i]['centroid'] - agent.pos
                    dist_c = float(np.linalg.norm(agent_to_c))
                    goal_delta = agent_goal - candidates[i]['centroid']
                    dist_goal = float(np.linalg.norm(goal_delta))
                    # v contains: [..., dist_goal, c.x, c.y, agent_to_c, dist_c, num_enemies, min_de, num_see_c, sees_agent]
                    num_enemies = v[12]
                    min_de = v[13]
                    num_see_c = v[14]
                    y = -0.5 * dist_goal - 0.1 * dist_c + 0.2 * float(min_de) - 1.0 * float(num_see_c)

                feats.append(v.astype(np.float32))
                targets.append(y)

            # Nudge agent to diversify state
            agent.pos = agent.pos + (agent_goal - agent.pos) * 0.05

    X = np.stack(feats, axis=0) if feats else np.zeros((0, 16), dtype=np.float32)
    y = np.array(targets, dtype=np.float32) if targets else np.zeros((0,), dtype=np.float32)
    return X, y


def train_lstm_hl(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> LSTMQRegressor:
    input_dim = X.shape[1]
    model = LSTMQRegressor(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.float().to(device)
            yb = yb.float().to(device)
            # Tile to sequence length k
            xseq = torch.tile(xb.unsqueeze(1), (1, k, 1))
            q_pred_seq = model(xseq)  # (B, k, 1)
            q_last = q_pred_seq[:, -1, 0]
            loss = loss_fn(q_last, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    return model


def main():
    parser = argparse.ArgumentParser(description='Train LSTM high-level (HL) + MPC via distillation or heuristic.')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--steps-per-episode', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--teacher-dtqn', type=str, default='')
    parser.add_argument('--out', type=str, default='lstm_hl_mpc_checkpoint.pt')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = SimulationConfig()

    teacher_path = os.path.join(os.path.dirname(__file__), args.teacher_dtqn) if args.teacher_dtqn else ''
    X, y = collect_dataset(
        num_envs=args.episodes,
        steps_per_env=args.steps_per_episode,
        cfg=cfg,
        teacher_dtqn_ckpt=teacher_path,
        rng_seed=args.seed,
    )

    if X.shape[0] == 0:
        print('[train_lstm_hl_mpc] No data collected. Aborting.')
        return

    model = train_lstm_hl(
        X=X,
        y=y,
        k=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    out_path = os.path.join(os.path.dirname(__file__), args.out)
    torch.save(model.state_dict(), out_path)
    print(f'[train_lstm_hl_mpc] Saved checkpoint to {out_path}')


if __name__ == '__main__':
    main()


