#!/usr/bin/env python3
"""
Collect an offline dataset for CQL training.

Runs episodes with one or more behavioural policies (visibility_greedy,
low_level_only, dwa_fov, vfh_plus_fov) and records transitions as the
EnCoMP state representation (maps + position) together with EnCoMP-style
rewards (cover, threat, goal-progress, collision).

Usage
-----
    python -m encomp.code.collect_data --episodes 200 --out encomp/code/dataset.npz
"""

import argparse
import math
import os
import sys
import random

import numpy as np
import torch
from shapely.geometry import Point

# Ensure fov/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sim_core import (
    SimulationConfig, Agent, EnemyAgent,
    generate_enemies, generate_obstacles, add_goal_as_obstacle,
    normalize, desired_force, obstacle_force,
    anticipatory_enemy_avoidance_force, enemy_avoidance_force,
    any_enemy_sees, compute_predicted_fov_union,
    compute_visibility_polygon_raycast,
)
from encomp.code.maps import (
    generate_maps, generate_cover_map, generate_goal_map,
    force_to_action, action_to_force, GRID_SIZE, NUM_ACTIONS,
)


# Reward weights from the EnCoMP paper
LAMBDA_COVER = 0.5
LAMBDA_THREAT = 1.5
LAMBDA_GOAL = 1.0
LAMBDA_COLLISION = 5.0
GOAL_BONUS = 50.0


def encomp_reward(
    agent_pos: np.ndarray,
    prev_pos: np.ndarray,
    goal: np.ndarray,
    cover_val: float,
    in_enemy_fov: bool,
    collided: bool,
    reached_goal: bool,
) -> float:
    """Compute the four-component EnCoMP reward."""
    r_cover = LAMBDA_COVER * cover_val
    r_threat = -LAMBDA_THREAT if in_enemy_fov else 0.0
    prev_dist = np.linalg.norm(prev_pos - goal)
    curr_dist = np.linalg.norm(agent_pos - goal)
    r_goal = LAMBDA_GOAL * (prev_dist - curr_dist)
    r_collision = -LAMBDA_COLLISION if collided else 0.0
    r_bonus = GOAL_BONUS if reached_goal else 0.0
    return r_cover + r_threat + r_goal + r_collision + r_bonus


def _get_cover_at_pos(pos: np.ndarray, cover_map: np.ndarray, scene_min: float, scene_max: float) -> float:
    span = scene_max - scene_min
    gx = int((pos[0] - scene_min) / span * cover_map.shape[1])
    gy = int((pos[1] - scene_min) / span * cover_map.shape[0])
    gx = np.clip(gx, 0, cover_map.shape[1] - 1)
    gy = np.clip(gy, 0, cover_map.shape[0] - 1)
    return float(cover_map[gy, gx])


def _behavioural_force(
    method: str, agent: Agent, enemies, obstacles_shapely, agent_goal, cfg,
) -> np.ndarray:
    """Compute the control force for a given behavioural policy (simplified)."""
    if method == 'low_level_only':
        agent.goal = agent_goal.copy()
        F_d = desired_force(agent, desired_speed=cfg.desired_speed) * 10.0
        F_o = obstacle_force(agent, obstacles_shapely, threshold=cfg.obstacle_threshold, repulsive_coeff=cfg.obstacle_repulsive)
        F_e = np.zeros(2)
        F_a = np.zeros(2)
        pred_union = compute_predicted_fov_union(enemies, obstacles_shapely, T=cfg.exposure_T_pred)
        if not pred_union.is_empty:
            from shapely.geometry import Point as Pt
            p_pt = Pt(agent.pos)
            if pred_union.contains(p_pt):
                nearest = pred_union.boundary.interpolate(pred_union.boundary.project(p_pt))
                F_e += normalize(agent.pos - np.array([nearest.x, nearest.y])) * 500.0
        for e in enemies:
            F_a += anticipatory_enemy_avoidance_force(agent, e, T=cfg.exposure_T_pred, weight=cfg.anticipate_weight)
        return F_d + F_o + F_e + F_a

    elif method == 'visibility_greedy':
        headings = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        best_score = float('inf')
        best_dir = np.zeros(2)
        for ang in headings:
            dvec = np.array([np.cos(ang), np.sin(ang)])
            cand = agent.pos + dvec * cfg.desired_speed * cfg.dt
            dist_g = np.linalg.norm(agent_goal - cand)
            exp = 1.0 if any_enemy_sees(cand, enemies, obstacles_shapely) else 0.0
            clr = 0.0
            for obs in obstacles_shapely:
                d = Point(cand).distance(obs)
                if d < cfg.obstacle_threshold:
                    clr += (cfg.obstacle_threshold - d) * 50.0
            score = dist_g + 5.0 * exp + clr
            if score < best_score:
                best_score = score
                best_dir = dvec
        return best_dir * 10.0

    elif method == 'dwa_fov':
        v_samples = np.linspace(0.0, 2.0, 5)
        w_samples = np.linspace(-1.5, 1.5, 7)
        h_steps = max(1, int(1.0 / cfg.dt))
        best_cost = float('inf')
        best_v, best_w = 0.0, 0.0
        heading = normalize(agent.velocity) if np.linalg.norm(agent.velocity) > 1e-3 else normalize(agent_goal - agent.pos)
        for v in v_samples:
            for w in w_samples:
                pos = agent.pos.copy()
                theta = math.atan2(heading[1], heading[0])
                cost = 0.0
                for _ in range(h_steps):
                    theta += w * cfg.dt
                    pos += np.array([np.cos(theta), np.sin(theta)]) * v * cfg.dt
                    dg = np.linalg.norm(agent_goal - pos)
                    op = sum((cfg.obstacle_threshold - Point(pos).distance(o)) * 200.0
                             for o in obstacles_shapely if Point(pos).distance(o) < cfg.obstacle_threshold)
                    ep = 5.0 if any_enemy_sees(pos, enemies, obstacles_shapely) else 0.0
                    cost += dg + op + ep
                if cost < best_cost:
                    best_cost = cost
                    best_v, best_w = v, w
        theta0 = math.atan2(heading[1], heading[0]) + best_w * cfg.dt
        return np.array([np.cos(theta0), np.sin(theta0)]) * best_v * 2.0

    else:
        # Fallback: move toward goal
        agent.goal = agent_goal.copy()
        return desired_force(agent, desired_speed=cfg.desired_speed) * 10.0


def collect_episode(
    method: str,
    seed: int,
    cfg: SimulationConfig,
):
    """Run one episode and return list of transition dicts."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    steps = int(cfg.sim_time / cfg.dt)
    agent_start = np.array([cfg.scene_min, cfg.scene_min], dtype=float)
    agent_goal = np.array([cfg.scene_max, cfg.scene_max], dtype=float)
    agent = Agent(agent_start, agent_goal, max_speed=1.5)
    enemies = generate_enemies(cfg, rng)
    obstacles, obstacles_shapely = generate_obstacles(cfg, rng)
    add_goal_as_obstacle(obstacles, obstacles_shapely, agent_goal)

    cover_map = generate_cover_map(obstacles_shapely, cfg.scene_min, cfg.scene_max)
    goal_map = generate_goal_map(agent_goal, cfg.scene_min, cfg.scene_max)

    transitions = []
    prev_pos = agent.pos.copy()

    for t in range(steps):
        goal_dist = float(np.linalg.norm(agent.pos - agent_goal))
        reached = goal_dist <= cfg.goal_reach_dist

        maps_s, pos_s = generate_maps(
            agent.pos, agent_goal, enemies, obstacles_shapely,
            cfg.scene_min, cfg.scene_max,
            precomputed_cover=cover_map, precomputed_goal=goal_map,
        )

        if reached:
            in_fov = any_enemy_sees(agent.pos, enemies, obstacles_shapely) if enemies else False
            collided = any(Point(agent.pos).within(o) for o in obstacles_shapely)
            cov = _get_cover_at_pos(agent.pos, cover_map, cfg.scene_min, cfg.scene_max)
            r = encomp_reward(agent.pos, prev_pos, agent_goal, cov, in_fov, collided, True)
            transitions.append({
                'maps': maps_s, 'pos': pos_s, 'action': 8,
                'reward': r, 'next_maps': maps_s, 'next_pos': pos_s, 'done': True,
            })
            break

        force = _behavioural_force(method, agent, enemies, obstacles_shapely, agent_goal, cfg)
        action = force_to_action(force)
        prev_pos = agent.pos.copy()

        smoothing = 0.5 if method == 'visibility_greedy' else 0.9
        max_spd = 1.0 if method == 'visibility_greedy' else 1.5
        agent.update(force, cfg.dt, smoothing=smoothing, max_speed=max_spd)

        for e in enemies:
            e.update(cfg.dt, obstacles_shapely)

        in_fov = any_enemy_sees(agent.pos, enemies, obstacles_shapely) if enemies else False
        collided = any(Point(agent.pos).within(o) for o in obstacles_shapely)
        cov = _get_cover_at_pos(agent.pos, cover_map, cfg.scene_min, cfg.scene_max)
        r = encomp_reward(agent.pos, prev_pos, agent_goal, cov, in_fov, collided, False)

        maps_sp, pos_sp = generate_maps(
            agent.pos, agent_goal, enemies, obstacles_shapely,
            cfg.scene_min, cfg.scene_max,
            precomputed_cover=cover_map, precomputed_goal=goal_map,
        )

        transitions.append({
            'maps': maps_s, 'pos': pos_s, 'action': action,
            'reward': r, 'next_maps': maps_sp, 'next_pos': pos_sp, 'done': False,
        })

    return transitions


def main():
    parser = argparse.ArgumentParser(description='Collect offline dataset for EnCoMP CQL')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--methods', nargs='+', default=['visibility_greedy', 'low_level_only', 'dwa_fov'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str, default='encomp/code/dataset.npz')
    args = parser.parse_args()

    cfg = SimulationConfig()

    all_maps, all_pos, all_actions = [], [], []
    all_rewards, all_next_maps, all_next_pos, all_dones = [], [], [], []
    total_transitions = 0
    ep_idx = 0

    eps_per_method = args.episodes // len(args.methods)

    for method in args.methods:
        print(f"\nCollecting {eps_per_method} episodes with '{method}'...")
        for i in range(eps_per_method):
            seed = args.seed + ep_idx
            transitions = collect_episode(method, seed, cfg)
            for tr in transitions:
                all_maps.append(tr['maps'])
                all_pos.append(tr['pos'])
                all_actions.append(tr['action'])
                all_rewards.append(tr['reward'])
                all_next_maps.append(tr['next_maps'])
                all_next_pos.append(tr['next_pos'])
                all_dones.append(tr['done'])
            total_transitions += len(transitions)
            tag = 'DONE' if (transitions and transitions[-1]['done']) else 'TIMEOUT'
            ep_idx += 1
            if (i + 1) % 10 == 0 or i == eps_per_method - 1:
                print(f"  [{i+1}/{eps_per_method}] transitions so far: {total_transitions}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(
        args.out,
        maps=np.array(all_maps, dtype=np.float32),
        pos=np.array(all_pos, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.int64),
        rewards=np.array(all_rewards, dtype=np.float32),
        next_maps=np.array(all_next_maps, dtype=np.float32),
        next_pos=np.array(all_next_pos, dtype=np.float32),
        dones=np.array(all_dones, dtype=np.float32),
    )
    print(f"\nSaved {total_transitions} transitions to {args.out}")


if __name__ == '__main__':
    main()
