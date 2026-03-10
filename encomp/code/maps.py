"""
Multi-map perception for the EnCoMP baseline.

Generates three grid-based maps from the simulation state, following the
EnCoMP paper's multi-map perception pipeline adapted for the 2D simulation:

  1. Cover map  – obstacle proximity → cover density  (higher near obstacles)
  2. Threat map – enemy visibility   → threat level   (higher in enemy FOVs)
  3. Goal map   – distance to goal   → goal relevance (higher near goal)

The maps are returned as a (3, H, W) numpy array along with a (4,) position
vector [agent_x, agent_y, goal_x, goal_y], all normalised to roughly [0, 1].
"""

import numpy as np
from shapely.geometry import Point
from shapely.prepared import prep
from typing import List, Tuple

GRID_SIZE = 20
NUM_ACTIONS = 9

_DIAG = 1.0 / np.sqrt(2.0)
ACTION_DIRS = np.array([
    [0.0, 1.0],          # 0 – N
    [_DIAG, _DIAG],      # 1 – NE
    [1.0, 0.0],          # 2 – E
    [_DIAG, -_DIAG],     # 3 – SE
    [0.0, -1.0],         # 4 – S
    [-_DIAG, -_DIAG],    # 5 – SW
    [-1.0, 0.0],         # 6 – W
    [-_DIAG, _DIAG],     # 7 – NW
    [0.0, 0.0],          # 8 – wait
], dtype=np.float32)


def action_to_force(action_idx: int, speed: float = 1.0) -> np.ndarray:
    return ACTION_DIRS[action_idx] * speed


def _cell_centres(scene_min: float, scene_max: float, grid_size: int) -> Tuple[np.ndarray, np.ndarray, float]:
    cell_size = (scene_max - scene_min) / grid_size
    xs = np.linspace(scene_min + cell_size / 2, scene_max - cell_size / 2, grid_size)
    ys = np.linspace(scene_min + cell_size / 2, scene_max - cell_size / 2, grid_size)
    return xs, ys, cell_size


def _cell_grid(scene_min: float, scene_max: float, grid_size: int) -> np.ndarray:
    """Return (grid_size, grid_size, 2) array of cell centre coordinates."""
    xs, ys, _ = _cell_centres(scene_min, scene_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    return np.stack([xx, yy], axis=-1)


def generate_cover_map(
    obstacles_shapely: list,
    scene_min: float,
    scene_max: float,
    grid_size: int = GRID_SIZE,
    cover_range: float = 2.0,
) -> np.ndarray:
    """Cover density per cell: 1.0 inside/touching obstacle, decays with distance."""
    xs, ys, _ = _cell_centres(scene_min, scene_max, grid_size)
    cover = np.zeros((grid_size, grid_size), dtype=np.float32)
    prepped = [prep(obs) for obs in obstacles_shapely]
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            pt = Point(x, y)
            min_dist = float('inf')
            for k, obs in enumerate(obstacles_shapely):
                if prepped[k].contains(pt):
                    min_dist = 0.0
                    break
                d = pt.distance(obs)
                if d < min_dist:
                    min_dist = d
            cover[i, j] = max(0.0, 1.0 - min_dist / cover_range)
    return cover


def generate_threat_map_fast(
    enemies: list,
    scene_min: float,
    scene_max: float,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    Fast angular/distance threat approximation (no raycasting).

    For each cell, checks whether it falls within any enemy's primary or
    secondary FOV cone and is within range.  This avoids the expensive
    per-cell Shapely visibility-polygon containment check.
    """
    if not enemies:
        return np.zeros((grid_size, grid_size), dtype=np.float32)

    grid = _cell_grid(scene_min, scene_max, grid_size)  # (H, W, 2)
    threat = np.zeros((grid_size, grid_size), dtype=np.float32)

    for e in enemies:
        diff = grid - e.pos[np.newaxis, np.newaxis, :]  # (H, W, 2)
        dists = np.linalg.norm(diff, axis=-1)            # (H, W)
        dists_safe = np.where(dists < 1e-9, 1e-9, dists)
        unit = diff / dists_safe[..., np.newaxis]

        cos_angle = unit[..., 0] * e.direction[0] + unit[..., 1] * e.direction[1]
        half_cos = np.cos(e.fov / 2.0)

        in_primary = (cos_angle >= half_cos) & (dists <= e.view_range)
        secondary_range = e.view_range * e.secondary_range_factor
        in_secondary = (~(cos_angle >= half_cos)) & (dists <= secondary_range)

        visible = in_primary | in_secondary
        # Threat decays with distance
        t_val = np.where(visible, np.clip(1.0 - dists / e.view_range, 0, 1), 0.0)
        threat = np.maximum(threat, t_val)

    return threat.astype(np.float32)


def generate_goal_map(
    goal: np.ndarray,
    scene_min: float,
    scene_max: float,
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """Normalised proximity to goal: 1.0 at goal, 0.0 at max distance."""
    xs, ys, _ = _cell_centres(scene_min, scene_max, grid_size)
    gx, gy = float(goal[0]), float(goal[1])
    xx, yy = np.meshgrid(xs, ys)
    dists = np.sqrt((xx - gx) ** 2 + (yy - gy) ** 2)
    max_dist = np.sqrt(2.0) * (scene_max - scene_min)
    goal_map = 1.0 - (dists / max_dist)
    return goal_map.astype(np.float32)


def generate_maps(
    agent_pos: np.ndarray,
    goal_pos: np.ndarray,
    enemies: list,
    obstacles_shapely: list,
    scene_min: float,
    scene_max: float,
    grid_size: int = GRID_SIZE,
    precomputed_cover: np.ndarray = None,
    precomputed_goal: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full state encoding.

    Returns
    -------
    maps : ndarray, shape (3, grid_size, grid_size)
        Channel 0 = cover, 1 = threat, 2 = goal.
    pos_vec : ndarray, shape (4,)
        [agent_x, agent_y, goal_x, goal_y] normalised to [0, 1].
    """
    if precomputed_cover is not None:
        cover = precomputed_cover
    else:
        cover = generate_cover_map(obstacles_shapely, scene_min, scene_max, grid_size)

    threat = generate_threat_map_fast(enemies, scene_min, scene_max, grid_size)

    if precomputed_goal is not None:
        goal_map = precomputed_goal
    else:
        goal_map = generate_goal_map(goal_pos, scene_min, scene_max, grid_size)

    maps = np.stack([cover, threat, goal_map], axis=0)

    span = scene_max - scene_min
    pos_vec = np.array([
        (agent_pos[0] - scene_min) / span,
        (agent_pos[1] - scene_min) / span,
        (goal_pos[0] - scene_min) / span,
        (goal_pos[1] - scene_min) / span,
    ], dtype=np.float32)

    return maps, pos_vec


def force_to_action(force: np.ndarray, threshold: float = 0.05) -> int:
    """Discretise a 2-D force vector into the nearest of the 9 actions."""
    mag = np.linalg.norm(force)
    if mag < threshold:
        return 8  # wait
    unit = force / mag
    dots = ACTION_DIRS[:8] @ unit
    return int(np.argmax(dots))
