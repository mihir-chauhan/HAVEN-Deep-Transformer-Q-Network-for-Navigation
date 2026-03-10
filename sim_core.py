import os
import math
import random
import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely.prepared import prep

import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.animation as animation

from dtqn_model import DTQN, ObservationHistory, RunningNormalizer
from encomp.code.maps import (
    generate_maps as encomp_generate_maps,
    generate_cover_map as encomp_cover_map,
    generate_goal_map as encomp_goal_map,
    action_to_force as encomp_action_to_force,
    GRID_SIZE as ENCOMP_GRID,
)
from encomp.code.model import EnCompCQLAgent
class LSTMQRegressor(nn.Module):
    """
    Simple LSTM-based regressor mapping a sequence of feature vectors to a scalar Q.
    Input shape: (batch, seq_len, input_dim)
    Output: (batch, seq_len, 1) with last-step used as scalar.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y, _ = self.lstm(x)
        out = self.head(y)
        return out


class LSTMPointRegressor(nn.Module):
    """
    LSTM-based regressor that outputs a 2D vector (e.g., desired waypoint delta) per step.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.lstm(x)
        out = self.head(y)
        return out


# --------------------------
# Configuration
# --------------------------
class SimulationConfig:
    """
    Container for simulation configuration.
    """
    def __init__(
        self,
        scene_min: float = -5.0,
        scene_max: float = 15.0,
        dt: float = 0.25,
        sim_time: float = 250.0,
        num_enemies: int = 5,
        num_obstacles: int = 6,
        enemy_speed: float = 0.3,
        enemy_fov_rad: float = math.pi / 3,
        secondary_fov_factor: float = 0.3,
        max_k_per_option: int = 20,
        gamma: float = 0.99,
        exposure_T_pred: float = 10.0,
        anticipate_weight: float = 1500.0,
        enemy_avoid_weight: float = 300.0,
        desired_speed: float = 1.0,
        obstacle_threshold: float = 1.0,
        obstacle_repulsive: float = 100.0,
        wait_probe_dist: float = 0.5,
        proximity_risk_dist: float = 2.0,
        goal_reach_dist: float = 2.0,
        # MPC-related parameters
        mpc_horizon_s: float = 1.2,
        mpc_num_samples: int = 64,
        mpc_speed_limit: float = 2.0,
        mpc_lambda_goal: float = 1.0,
        mpc_lambda_exposure: float = 10.0,
        mpc_lambda_obstacle: float = 200.0,
        mpc_collision_penalty: float = 5000.0,
        mpc_clearance_thresh: float = 1.0,
        mpc_smoothness_penalty: float = 0.1,
        # Raycasting fidelity
        fov_num_rays: int = 24,
    ):
        self.scene_min = scene_min
        self.scene_max = scene_max
        self.dt = dt
        self.sim_time = sim_time
        self.num_enemies = num_enemies
        self.num_obstacles = num_obstacles
        self.enemy_speed = enemy_speed
        self.enemy_fov_rad = enemy_fov_rad
        self.secondary_fov_factor = secondary_fov_factor
        self.max_k_per_option = max_k_per_option
        self.gamma = gamma
        self.exposure_T_pred = exposure_T_pred
        self.anticipate_weight = anticipate_weight
        self.enemy_avoid_weight = enemy_avoid_weight
        self.desired_speed = desired_speed
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_repulsive = obstacle_repulsive
        self.wait_probe_dist = wait_probe_dist
        self.proximity_risk_dist = proximity_risk_dist
        self.goal_reach_dist = goal_reach_dist
        # MPC params
        self.mpc_horizon_s = mpc_horizon_s
        self.mpc_num_samples = mpc_num_samples
        self.mpc_speed_limit = mpc_speed_limit
        self.mpc_lambda_goal = mpc_lambda_goal
        self.mpc_lambda_exposure = mpc_lambda_exposure
        self.mpc_lambda_obstacle = mpc_lambda_obstacle
        self.mpc_collision_penalty = mpc_collision_penalty
        self.mpc_clearance_thresh = mpc_clearance_thresh
        self.mpc_smoothness_penalty = mpc_smoothness_penalty
        # Raycasting fidelity
        self.fov_num_rays = fov_num_rays


# --------------------------
# Utilities
# --------------------------
def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def nearest_points_on_polygon(point: np.ndarray, polygon: Polygon) -> List[Tuple[float, float]]:
    point_geom = Point(point)
    if polygon.contains(point_geom):
        return [(point[0], point[1]), (point[0], point[1])]
    boundary = polygon.boundary
    nearest_point = boundary.interpolate(boundary.project(point_geom))
    return [(nearest_point.x, nearest_point.y), (point[0], point[1])]


def obstacle_force(entity, obstacles: List[Polygon], threshold: float = 1.0, repulsive_coeff: float = 100.0) -> np.ndarray:
    force = np.zeros(2)
    entity_point = Point(entity.pos)
    for obs in obstacles:
        if entity_point.distance(obs) < threshold:
            nearest_points = nearest_points_on_polygon(entity.pos, obs)
            if nearest_points:
                nearest_point = np.array(nearest_points[0])
                d = np.linalg.norm(entity.pos - nearest_point)
                if d < threshold and d > 1e-6:
                    force += repulsive_coeff * (1.0 / d - 1.0 / threshold) * normalize(entity.pos - nearest_point)
    return force


def desired_force(agent, desired_speed: float = 1.0) -> np.ndarray:
    diff = agent.goal - agent.pos
    return desired_speed * normalize(diff)


def compute_visibility_polygon_raycast(
    pos: np.ndarray,
    direction: np.ndarray,
    fov: float,
    view_range: float,
    obstacles: List[Polygon],
    num_rays: int = 50,
    secondary_view_range_factor: float = 0.3,
) -> Polygon:
    points = []
    secondary_range = view_range * secondary_view_range_factor
    primary_rays = num_rays
    rel_angles_primary = np.linspace(-fov / 2, fov / 2, primary_rays)
    for angle in rel_angles_primary:
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ray_dir = normalize(np.dot(rot_matrix, direction))
        end_point = pos + view_range * ray_dir
        ray = LineString([pos, end_point])
        nearest_dist = view_range
        nearest_point = end_point
        for obs in obstacles:
            inter = ray.intersection(obs)
            if not inter.is_empty:
                if inter.geom_type == 'Point':
                    candidate = np.array(inter.coords[0])
                    d = np.linalg.norm(candidate - pos)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_point = candidate
                elif inter.geom_type == 'MultiPoint':
                    for pt in inter:
                        candidate = np.array(pt.coords[0])
                        d = np.linalg.norm(candidate - pos)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_point = candidate
                elif inter.geom_type == 'LineString':
                    candidate = np.array(inter.coords[0])
                    d = np.linalg.norm(candidate - pos)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_point = candidate
        points.append(nearest_point)

    secondary_rays = num_rays
    rel_angles_secondary = np.linspace(fov / 2, 2 * np.pi - fov / 2, secondary_rays)
    for angle in rel_angles_secondary:
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ray_dir = normalize(np.dot(rot_matrix, direction))
        end_point = pos + secondary_range * ray_dir
        ray = LineString([pos, end_point])
        nearest_dist = secondary_range
        nearest_point = end_point
        for obs in obstacles:
            inter = ray.intersection(obs)
            if not inter.is_empty:
                if inter.geom_type == 'Point':
                    candidate = np.array(inter.coords[0])
                    d = np.linalg.norm(candidate - pos)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_point = candidate
                elif inter.geom_type == 'MultiPoint':
                    for pt in inter:
                        candidate = np.array(pt.coords[0])
                        d = np.linalg.norm(candidate - pos)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_point = candidate
                elif inter.geom_type == 'LineString':
                    candidate = np.array(inter.coords[0])
                    d = np.linalg.norm(candidate - pos)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_point = candidate
        points.append(nearest_point)

    poly_coords = [pos] + points
    poly = Polygon(poly_coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        r = 0.1
        poly = Polygon([(pos[0]-r, pos[1]-r), (pos[0]+r, pos[1]-r),
                         (pos[0]+r, pos[1]+r), (pos[0]-r, pos[1]+r)])
    return poly


def predict_detection_probability(
    point: np.ndarray,
    enemy,
    T: float = 10.0,
    angle_threshold: float = 0.3,
    dist_threshold: float = 2.0,
    a: float = 10.0,
    b: float = 10.0,
) -> float:
    predicted_pos = enemy.pos + enemy.direction * enemy.speed * T
    v = point - predicted_pos
    d = np.linalg.norm(v)
    if d == 0:
        return 1.0
    v_norm = v / d
    angle_diff = np.arccos(np.clip(np.dot(v_norm, enemy.direction), -1.0, 1.0))
    p_angle = 1.0 / (1.0 + np.exp(a * (angle_diff - angle_threshold)))
    p_dist = 1.0 / (1.0 + np.exp(b * (d - dist_threshold)))
    return float(p_angle * p_dist)


def any_enemy_sees(point: np.ndarray, enemies: List["EnemyAgent"], obstacles: List[Polygon]) -> bool:
    for enemy in enemies:
        if enemy.can_see(point, obstacles):
            return True
    return False


def agent_inside_any_obstacle(point: np.ndarray, obstacles: List[Polygon]) -> bool:
    p = Point(point)
    for obs in obstacles:
        if obs.contains(p):
            return True
    return False


# --------------------------
# Entities
# --------------------------
class Agent:
    def __init__(self, pos: np.ndarray, goal: np.ndarray, max_speed: float = 1.0):
        self.pos = np.array(pos, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.velocity = np.zeros(2)
        self.max_speed = max_speed

    def update(self, force: np.ndarray, dt: float, smoothing: float = 0.9, max_speed: Optional[float] = None):
        self.velocity = smoothing * self.velocity + (1 - smoothing) * force
        speed = np.linalg.norm(self.velocity)
        cap = max_speed if max_speed is not None else self.max_speed
        if speed > cap:
            self.velocity = (self.velocity / speed) * cap
        self.pos += self.velocity * dt


class EnemyAgent:
    def __init__(
        self,
        pos: np.ndarray,
        direction: np.ndarray,
        fov: float,
        view_range: float,
        speed: float,
        secondary_range_factor: float,
        scene_min: float,
        scene_max: float,
        num_rays: int,
    ):
        self.pos = np.array(pos, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))
        self.fov = fov
        self.view_range = view_range
        self.speed = speed
        self.secondary_range_factor = secondary_range_factor
        self.scene_min = scene_min
        self.scene_max = scene_max
        self.num_rays = num_rays

    def update(self, dt: float, obstacles: List[Polygon]):
        F_obs = obstacle_force(self, obstacles, threshold=0.5, repulsive_coeff=200.0)
        self.direction = normalize(self.direction + 0.3 * F_obs)
        self.pos += self.direction * self.speed * dt
        if self.pos[0] < self.scene_min or self.pos[0] > self.scene_max:
            self.direction[0] = -self.direction[0]
        if self.pos[1] < self.scene_min or self.pos[1] > self.scene_max:
            self.direction[1] = -self.direction[1]
        angle = 0.01 * dt
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        self.direction = normalize(np.dot(rot, self.direction))

    def get_visibility_polygon(self, obstacles: List[Polygon]) -> Polygon:
        return compute_visibility_polygon_raycast(
            self.pos,
            self.direction,
            self.fov,
            self.view_range,
            obstacles,
            num_rays=self.num_rays,
            secondary_view_range_factor=self.secondary_range_factor,
        )

    def can_see(self, point: np.ndarray, obstacles: List[Polygon]) -> bool:
        """Fast single-ray visibility check (matches older_main.py logic)."""
        diff = np.asarray(point) - self.pos
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return True
        cos_angle = np.dot(self.direction, diff / dist)
        half_fov_cos = np.cos(self.fov / 2)
        if cos_angle >= half_fov_cos:
            effective_range = self.view_range
        else:
            effective_range = self.view_range * self.secondary_range_factor
        if dist > effective_range:
            return False
        ray = LineString([self.pos, point])
        for obs in obstacles:
            inter = ray.intersection(obs)
            if inter.is_empty:
                continue
            if inter.geom_type == 'Point':
                d = np.linalg.norm(np.array(inter.coords[0]) - self.pos)
            elif inter.geom_type == 'MultiPoint':
                d = min(np.linalg.norm(np.array(pt.coords[0]) - self.pos) for pt in inter.geoms)
            elif inter.geom_type == 'LineString':
                d = np.linalg.norm(np.array(inter.coords[0]) - self.pos)
            else:
                d = 0.0
            if d < dist - 1e-6:
                return False
        return True


# --------------------------
# Environment generation
# --------------------------
def create_random_convex_polygon(scene_min: float, scene_max: float, sides: int = 5) -> Polygon:
    sides = np.random.randint(3, sides + 1)
    center_x = np.random.uniform(scene_min, scene_max - 3)
    center_y = np.random.uniform(scene_min, scene_max - 3)
    radius = np.random.uniform(0.5, 2.0)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, sides))
    coords = []
    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append((x, y))
    return Polygon(coords)


def generate_enemies(cfg: SimulationConfig, rng: np.random.Generator) -> List[EnemyAgent]:
    enemies: List[EnemyAgent] = []
    for _ in range(cfg.num_enemies):
        pos = rng.uniform(cfg.scene_min + 2, cfg.scene_max, size=2)
        angle = rng.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        fov = cfg.enemy_fov_rad + rng.uniform(-0.1, 0.1)
        enemies.append(
            EnemyAgent(
                pos=pos,
                direction=direction,
                fov=fov,
                view_range=5.0,
                speed=cfg.enemy_speed,
                secondary_range_factor=cfg.secondary_fov_factor,
                scene_min=cfg.scene_min,
                scene_max=cfg.scene_max,
                num_rays=cfg.fov_num_rays,
            )
        )
    return enemies


def generate_obstacles(cfg: SimulationConfig, rng: np.random.Generator) -> Tuple[List[Tuple[Polygon, Tuple[float, float, float, float], np.ndarray]], List[Polygon]]:
    obstacles: List[Tuple[Polygon, Tuple[float, float, float, float], np.ndarray]] = []
    obstacles_shapely: List[Polygon] = []
    for _ in range(cfg.num_obstacles):
        poly = create_random_convex_polygon(cfg.scene_min, cfg.scene_max, sides=5)
        obstacles_shapely.append(poly)
        bounds = poly.bounds
        centroid = np.array([poly.centroid.x, poly.centroid.y])
        obstacles.append((poly, bounds, centroid))
    return obstacles, obstacles_shapely


def add_goal_as_obstacle(obstacles: List[Tuple[Polygon, Tuple[float, float, float, float], np.ndarray]], obstacles_shapely: List[Polygon], goal: np.ndarray) -> None:
    goal_poly = Polygon(
        [
            (goal[0] - 0.5, goal[1] - 0.5),
            (goal[0] + 0.5, goal[1] - 0.5),
            (goal[0] + 0.5, goal[1] + 0.5),
            (goal[0] - 0.5, goal[1] + 0.5),
        ]
    )
    obstacles_shapely.append(goal_poly)
    obstacles.append((goal_poly, goal_poly.bounds, np.array([goal_poly.centroid.x, goal_poly.centroid.y])))


# --------------------------
# Metrics and results
# --------------------------
class EpisodeResult:
    def __init__(self):
        self.success: bool = False
        self.steps_taken: int = 0
        self.path_length: float = 0.0
        self.time_to_goal: float = float('nan')
        self.collision: bool = False
        self.exposure_time: float = 0.0
        # Trajectory data (only populated when record_trajectory=True)
        self.agent_positions: Optional[List[np.ndarray]] = None
        self.enemy_positions: Optional[List[List[np.ndarray]]] = None  # [step][enemy_idx]
        self.enemy_directions: Optional[List[List[np.ndarray]]] = None
        self.obstacles_shapely_snapshot: Optional[List[Polygon]] = None
        self.obstacles_snapshot: Optional[List[Tuple]] = None
        self.goal: Optional[np.ndarray] = None
        self.num_enemies: int = 0
        self.enemy_fovs: Optional[List[float]] = None
        self.enemy_view_ranges: Optional[List[float]] = None


def save_episode_video(result: 'EpisodeResult', out_path: str, cfg: Optional['SimulationConfig'] = None, fps: int = 20, title: str = "Evaluation Episode") -> None:
    """Save an episode trajectory as an MP4 video from recorded EpisodeResult data."""
    if result.agent_positions is None or len(result.agent_positions) < 2:
        return
    agent_pos_arr = np.array(result.agent_positions)
    num_frames = len(agent_pos_arr)
    num_enemies = result.num_enemies
    sc_min = cfg.scene_min if cfg else -5.0
    sc_max = cfg.scene_max if cfg else 15.0

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(sc_min - 2, sc_max + 2)
    ax.set_ylim(sc_min - 2, sc_max + 2)
    ax.set_aspect('equal')
    ax.set_title(title)

    # Draw static obstacles (gray) and goal (green)
    obs_list = result.obstacles_snapshot or []
    goal = result.goal
    for obs_data in obs_list:
        poly = obs_data[0]
        coords = np.array(poly.exterior.coords)
        is_goal = (goal is not None and poly.contains(Point(goal)))
        color = 'green' if is_goal else 'gray'
        alpha = 0.7 if is_goal else 0.5
        ax.add_patch(MplPolygon(coords, closed=True, color=color, alpha=alpha))

    # Goal marker
    if goal is not None:
        ax.plot(goal[0], goal[1], 'g*', ms=15, zorder=10)

    # Start marker
    ax.plot(agent_pos_arr[0, 0], agent_pos_arr[0, 1], 'bs', ms=10, zorder=10, label='Start')

    # Initialize artists
    line, = ax.plot([], [], 'b-', lw=1.5, alpha=0.7)
    agent_dot, = ax.plot([], [], 'bo', ms=8, zorder=5)

    enemy_dots = []
    enemy_vis_patches = []
    for _ in range(num_enemies):
        dot, = ax.plot([], [], 'rs', ms=7)
        enemy_dots.append(dot)
        patch = MplPolygon([[0, 0]], closed=True, color='red', alpha=0.15)
        ax.add_patch(patch)
        enemy_vis_patches.append(patch)

    obstacles_shapely = result.obstacles_shapely_snapshot or []
    enemy_fovs = result.enemy_fovs or [math.pi / 3] * num_enemies
    enemy_view_ranges = result.enemy_view_ranges or [5.0] * num_enemies

    def init():
        line.set_data([], [])
        agent_dot.set_data([], [])
        for d in enemy_dots:
            d.set_data([], [])
        for p in enemy_vis_patches:
            p.set_xy([[0, 0]])
        return (line, agent_dot, *enemy_dots, *enemy_vis_patches)

    def animate(i):
        line.set_data(agent_pos_arr[:i + 1, 0], agent_pos_arr[:i + 1, 1])
        agent_dot.set_data([agent_pos_arr[i, 0]], [agent_pos_arr[i, 1]])
        if result.enemy_positions and i < len(result.enemy_positions):
            for j in range(num_enemies):
                epos = result.enemy_positions[i][j]
                edir = result.enemy_directions[i][j]
                enemy_dots[j].set_data([epos[0]], [epos[1]])
                vis_poly = compute_visibility_polygon_raycast(
                    epos, edir, enemy_fovs[j], enemy_view_ranges[j],
                    obstacles_shapely, num_rays=100
                )
                if hasattr(vis_poly, 'exterior'):
                    enemy_vis_patches[j].set_xy(np.array(vis_poly.exterior.coords))
                else:
                    enemy_vis_patches[j].set_xy([[0, 0]])
        return (line, agent_dot, *enemy_dots, *enemy_vis_patches)

    # Subsample long episodes to keep videos manageable (max ~400 frames)
    max_video_frames = 400
    if num_frames > max_video_frames:
        step = num_frames // max_video_frames
        frame_indices = list(range(0, num_frames, step))
    else:
        frame_indices = list(range(num_frames))

    ani = animation.FuncAnimation(fig, animate, frames=frame_indices, interval=50, blit=True, init_func=init)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    # Try to find ffmpeg binary (system or imageio-ffmpeg bundle)
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path
    except ImportError:
        pass
    saved = False
    # Try ffmpeg with various codecs
    for codec in ['libx264', 'mpeg4', 'libxvid', None]:
        try:
            extra = ['-vcodec', codec] if codec else []
            writer = animation.FFMpegWriter(fps=fps, extra_args=extra)
            ani.save(out_path, writer=writer)
            saved = True
            break
        except Exception:
            continue
    if not saved:
        # Fallback to pillow GIF
        gif_path = out_path.replace('.mp4', '.gif')
        try:
            ani.save(gif_path, writer='pillow', fps=min(fps, 10))
            print(f"  (saved as GIF instead: {gif_path})")
            saved = True
        except Exception as e:
            print(f"  Warning: all video save methods failed: {e}")
    plt.close(fig)
    if saved:
        print(f"  Video saved: {out_path}")


# --------------------------
# Control helpers
# --------------------------
def anticipatory_enemy_avoidance_force(agent: Agent, enemy: EnemyAgent, T: float, weight: float = 1500.0, epsilon: float = 0.1) -> np.ndarray:
    predicted_pos = enemy.pos + enemy.direction * enemy.speed * T
    v = agent.pos - predicted_pos
    d = np.linalg.norm(v)
    if d == 0:
        return np.zeros(2)
    p = predict_detection_probability(agent.pos, enemy, T=T)
    return normalize(v) * (weight * (p ** 5) / (d + epsilon))


def enemy_avoidance_force(agent: Agent, enemy: EnemyAgent, weight: float = 50.0, obstacles: Optional[List[Polygon]] = None, epsilon: float = 0.1) -> np.ndarray:
    if enemy.can_see(agent.pos, obstacles or []):
        diff = agent.pos - enemy.pos
        d = np.linalg.norm(diff)
        if d == 0:
            return np.zeros(2)
        return normalize(diff) * (weight / ((d + epsilon) ** 2))
    return np.zeros(2)


def compute_predicted_fov_union(enemies: List[EnemyAgent], obstacles: List[Polygon], T: float) -> Polygon:
    polys = []
    for e in enemies:
        predicted_pos = e.pos + e.direction * e.speed * T
        poly = compute_visibility_polygon_raycast(
            pos=predicted_pos,
            direction=e.direction,
            fov=e.fov,
            view_range=e.view_range,
            obstacles=obstacles,
            secondary_view_range_factor=e.secondary_range_factor,
        )
        polys.append(poly)
    return unary_union([p for p in polys if p.is_valid]) if polys else Polygon()


def simulate_dynamics(pos: np.ndarray, vel: np.ndarray, action: np.ndarray, dt: float, max_speed: float) -> Tuple[np.ndarray, np.ndarray]:
    new_vel = 0.9 * vel + 0.1 * action
    speed = np.linalg.norm(new_vel)
    if speed > max_speed:
        new_vel = (new_vel / speed) * max_speed
    new_pos = pos + new_vel * dt
    return new_pos, new_vel


def mpc_control_step(
    agent: Agent,
    target: np.ndarray,
    enemies: List[EnemyAgent],
    obstacles_shapely: List[Polygon],
    cfg: SimulationConfig,
) -> np.ndarray:
    """
    MPPI-style sampling-based MPC over constant actions per step.
    Returns the first action of the best sequence as a force-like vector.
    """
    horizon_steps = max(1, int(cfg.mpc_horizon_s / cfg.dt))
    num_samples = cfg.mpc_num_samples
    rng = np.random.default_rng()
    best_cost = float('inf')
    best_a0 = np.zeros(2)

    # Bias mean action toward target direction
    to_target = target - agent.pos
    if np.linalg.norm(to_target) < 1e-6:
        mean_dir = np.zeros(2)
    else:
        mean_dir = normalize(to_target)
    mean_action = mean_dir * cfg.mpc_speed_limit

    # Precompute current FoV union once for this control step and prepare for fast contains()
    if enemies:
        current_polys = [e.get_visibility_polygon(obstacles_shapely) for e in enemies]
        fov_union = unary_union(current_polys)
        fov_prepared = prep(fov_union) if not fov_union.is_empty else None
    else:
        fov_prepared = None

    for _ in range(num_samples):
        # Sample noisy action sequence around mean
        actions = mean_action + 0.5 * rng.normal(size=(horizon_steps, 2))
        pos = agent.pos.copy()
        vel = agent.velocity.copy()
        cost = 0.0
        prev_a = None
        for t in range(horizon_steps):
            a = actions[t]
            pos, vel = simulate_dynamics(pos, vel, a, cfg.dt, cfg.mpc_speed_limit)
            p_pt = Point(pos)
            # Collision/obstacle clearance cost
            clearance_pen = 0.0
            collided = False
            for obs in obstacles_shapely:
                if obs.contains(p_pt):
                    collided = True
                    break
                d = p_pt.distance(obs)
                if d < cfg.mpc_clearance_thresh:
                    clearance_pen += (cfg.mpc_clearance_thresh - d) * cfg.mpc_lambda_obstacle
            if collided:
                cost += cfg.mpc_collision_penalty
                break
            # Exposure cost
            if fov_prepared is not None and fov_prepared.contains(p_pt):
                exposure = 1.0
            else:
                exposure = 0.0
            # Goal cost (to target)
            dist_target = np.linalg.norm(target - pos)
            # Smoothness
            smooth_pen = 0.0
            if prev_a is not None:
                smooth_pen = cfg.mpc_smoothness_penalty * np.linalg.norm(a - prev_a)
            prev_a = a
            cost += cfg.mpc_lambda_goal * dist_target + cfg.mpc_lambda_exposure * exposure + clearance_pen + smooth_pen
            # Early pruning if already worse than current best
            if cost >= best_cost:
                break
        if cost < best_cost:
            best_cost = cost
            best_a0 = actions[0]
    return best_a0


# --------------------------
# Methods / Controllers
# --------------------------
class DTQNHighLevelPolicy:
    def __init__(self, k: int = 3, eps_greedy: float = 0.05, lr: float = 1e-3, checkpoint_path: Optional[str] = None, device: Optional[str] = None,
                 hysteresis_bonus: float = 0.15):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.eps = eps_greedy
        self.model: Optional[DTQN] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.input_dim: Optional[int] = None
        self.criterion = nn.MSELoss()
        self.checkpoint_path = checkpoint_path
        self.normalizer: Optional[RunningNormalizer] = None
        self.obs_history: Optional[ObservationHistory] = None
        # Subgoal commitment: bonus added to Q-value of current subgoal
        self.hysteresis_bonus = hysteresis_bonus
        self.current_subgoal_idx: Optional[int] = None

    def _ensure_model(self, input_dim: int):
        if self.model is None:
            self.input_dim = input_dim
            self.model = DTQN(input_dim=input_dim, output_dim=1, k=self.k).to(self.device)
            self.normalizer = RunningNormalizer(dim=input_dim)
            self.obs_history = ObservationHistory(k=self.k, dim=input_dim)
            if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
                ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    self.model.load_state_dict(ckpt['model'])
                    if 'normalizer' in ckpt:
                        self.normalizer.load_state_dict(ckpt['normalizer'])
                    if 'eps' in ckpt:
                        self.eps = min(self.eps, ckpt['eps'])  # use eval eps (lower)
                else:
                    self.model.load_state_dict(ckpt)
            self.model.eval()

    def encode_state_candidate(self, agent: Agent, enemies: List[EnemyAgent], candidate: Dict, goal: np.ndarray, obstacles_shapely: List[Polygon]) -> np.ndarray:
        agent_pos = agent.pos
        agent_vel = agent.velocity
        goal_delta = goal - agent.pos
        dist_goal = np.linalg.norm(goal_delta)
        c = candidate['centroid']
        agent_to_c = c - agent.pos
        dist_c = np.linalg.norm(agent_to_c)
        num_enemies = float(len(enemies))
        min_de = float('inf')
        num_see_c = 0.0
        for e in enemies:
            de = np.linalg.norm(c - e.pos)
            if de < min_de:
                min_de = de
            if e.can_see(c, obstacles_shapely):
                num_see_c += 1.0
        if min_de == float('inf'):
            min_de = 0.0
        sees_agent = 1.0 if any_enemy_sees(agent.pos, enemies, obstacles_shapely) else 0.0
        raw = np.array([
            agent_pos[0], agent_pos[1],
            agent_vel[0], agent_vel[1],
            goal_delta[0], goal_delta[1],
            dist_goal,
            c[0], c[1],
            agent_to_c[0], agent_to_c[1],
            dist_c,
            num_enemies,
            min_de,
            num_see_c,
            sees_agent,
        ], dtype=np.float32)
        self._ensure_model(len(raw))
        return self.normalizer.normalize(raw)

    def select_subgoal(self, agent: Agent, enemies: List[EnemyAgent], candidates: List[Dict], mask: np.ndarray, goal: np.ndarray, obstacles_shapely: List[Polygon]) -> int:
        valid_indices = [i for i, m in enumerate(mask) if m]
        if not valid_indices:
            return -1
        q_values: List[float] = []
        cand_indices: List[int] = []
        feature_vecs: List[np.ndarray] = []
        seqs: List[np.ndarray] = []
        for i in valid_indices:
            v = self.encode_state_candidate(agent, enemies, candidates[i], goal, obstacles_shapely)
            seq = self.obs_history.get_sequence(v)
            cand_indices.append(i)
            feature_vecs.append(v)
            seqs.append(seq)
        # Batch forward pass
        with torch.no_grad():
            batch = torch.from_numpy(np.stack(seqs, axis=0)).float().to(self.device)
            q_all = self.model(batch)[:, -1, 0]
            q_values = q_all.cpu().tolist()
        if np.random.rand() < self.eps:
            chosen_local = int(np.random.randint(len(cand_indices)))
        else:
            # Apply hysteresis: bonus Q for current subgoal to reduce switching
            if self.hysteresis_bonus > 0 and self.current_subgoal_idx is not None:
                adjusted_q = list(q_values)
                for j, ci in enumerate(cand_indices):
                    if ci == self.current_subgoal_idx:
                        adjusted_q[j] += self.hysteresis_bonus
                        break
                chosen_local = int(np.argmax(adjusted_q))
            else:
                chosen_local = int(np.argmax(q_values))
        # Push chosen feature vector into observation history for temporal context
        self.obs_history.push(feature_vecs[chosen_local])
        self.current_subgoal_idx = cand_indices[chosen_local]
        return cand_indices[chosen_local]


class LSTMHighLevelPolicy:
    def __init__(self, k: int = 3, eps_greedy: float = 0.1, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.eps = eps_greedy
        self.model: Optional[LSTMQRegressor] = None
        self.input_dim: Optional[int] = None
        self.checkpoint_path = checkpoint_path

    def _ensure_model(self, input_dim: int):
        if self.model is None:
            self.input_dim = input_dim
            self.model = LSTMQRegressor(input_dim=input_dim).to(self.device)
            if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
                try:
                    self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
                    self.model.eval()
                except Exception:
                    pass

    def encode_state_candidate(self, agent: Agent, enemies: List[EnemyAgent], candidate: Dict, goal: np.ndarray, obstacles_shapely: List[Polygon]) -> np.ndarray:
        agent_pos = agent.pos
        agent_vel = agent.velocity
        goal_delta = goal - agent.pos
        dist_goal = np.linalg.norm(goal_delta)
        c = candidate['centroid']
        agent_to_c = c - agent.pos
        dist_c = np.linalg.norm(agent_to_c)
        num_enemies = float(len(enemies))
        min_de = float('inf')
        num_see_c = 0.0
        for e in enemies:
            de = np.linalg.norm(c - e.pos)
            if de < min_de:
                min_de = de
            if e.can_see(c, obstacles_shapely):
                num_see_c += 1.0
        if min_de == float('inf'):
            min_de = 0.0
        sees_agent = 1.0 if any_enemy_sees(agent.pos, enemies, obstacles_shapely) else 0.0
        vec = np.array([
            agent_pos[0], agent_pos[1],
            agent_vel[0], agent_vel[1],
            goal_delta[0], goal_delta[1],
            dist_goal,
            c[0], c[1],
            agent_to_c[0], agent_to_c[1],
            dist_c,
            num_enemies,
            min_de,
            num_see_c,
            sees_agent,
        ], dtype=np.float32)
        return vec

    def select_subgoal(self, agent: Agent, enemies: List[EnemyAgent], candidates: List[Dict], mask: np.ndarray, goal: np.ndarray, obstacles_shapely: List[Polygon]) -> int:
        valid_indices = [i for i, m in enumerate(mask) if m]
        if not valid_indices:
            return -1
        q_values: List[float] = []
        cand_indices: List[int] = []
        for i in valid_indices:
            v = self.encode_state_candidate(agent, enemies, candidates[i], goal, obstacles_shapely)
            self._ensure_model(len(v))
            with torch.no_grad():
                x = torch.from_numpy(np.tile(v, (self.k, 1))).float().to(self.device).unsqueeze(0)
                q = self.model(x)[:, -1, 0]
                q_values.append(q.item())
                cand_indices.append(i)
        if np.random.rand() < self.eps:
            return int(np.random.choice(cand_indices))
        best_idx = int(np.argmax(q_values))
        return cand_indices[best_idx]


class LSTMEnd2EndWaypointPolicy:
    """
    Outputs a local waypoint delta (dx, dy) that MPC will try to realize.
    """
    def __init__(self, k: int = 3, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.model: Optional[LSTMPointRegressor] = None
        self.input_dim: Optional[int] = None
        self.checkpoint_path = checkpoint_path

    def _ensure_model(self, input_dim: int):
        if self.model is None:
            self.input_dim = input_dim
            self.model = LSTMPointRegressor(input_dim=input_dim).to(self.device)
            if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
                try:
                    self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
                    self.model.eval()
                except Exception:
                    pass

    def encode_state(self, agent: Agent, goal: np.ndarray, enemies: List[EnemyAgent], obstacles_shapely: List[Polygon]) -> np.ndarray:
        goal_delta = goal - agent.pos
        dist_goal = np.linalg.norm(goal_delta)
        min_obs_dist = float('inf')
        for obs in obstacles_shapely:
            d = Point(agent.pos).distance(obs)
            if d < min_obs_dist:
                min_obs_dist = d
        num_enemies = float(len(enemies))
        sees_agent = 1.0 if any_enemy_sees(agent.pos, enemies, obstacles_shapely) else 0.0
        avg_enemy_dir = np.mean([e.direction for e in enemies], axis=0) if enemies else np.zeros(2)
        vec = np.array([
            agent.pos[0], agent.pos[1],
            agent.velocity[0], agent.velocity[1],
            goal_delta[0], goal_delta[1], dist_goal,
            min_obs_dist,
            num_enemies, sees_agent,
            avg_enemy_dir[0], avg_enemy_dir[1],
        ], dtype=np.float32)
        return vec

    def propose_local_waypoint(self, agent: Agent, goal: np.ndarray, enemies: List[EnemyAgent], obstacles_shapely: List[Polygon], max_radius: float = 3.0) -> np.ndarray:
        v = self.encode_state(agent, goal, enemies, obstacles_shapely)
        self._ensure_model(len(v))
        with torch.no_grad():
            x = torch.from_numpy(np.tile(v, (self.k, 1))).float().to(self.device).unsqueeze(0)
            out = self.model(x)[:, -1, :]  # (1,2)
            delta = out.squeeze(0).cpu().numpy()
        # Normalize and scale to max_radius; fallback to goal if zero
        if np.linalg.norm(delta) < 1e-6:
            dir_to_goal = goal - agent.pos
            if np.linalg.norm(dir_to_goal) < 1e-6:
                delta = np.zeros(2)
            else:
                delta = (dir_to_goal / np.linalg.norm(dir_to_goal)) * max_radius
        else:
            delta = (delta / (np.linalg.norm(delta) + 1e-8)) * max_radius
        return agent.pos + delta


class LSTMEnd2EndActionPolicy:
    """
    Outputs a 2D action (force-like vector) given recent state features.
    """
    def __init__(self, k: int = 3, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.model: Optional[LSTMPointRegressor] = None
        self.input_dim: Optional[int] = None
        self.checkpoint_path = checkpoint_path

    def _ensure_model(self, input_dim: int):
        if self.model is None:
            self.input_dim = input_dim
            self.model = LSTMPointRegressor(input_dim=input_dim).to(self.device)
            if self.checkpoint_path and os.path.isfile(self.checkpoint_path):
                try:
                    self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
                    self.model.eval()
                except Exception:
                    pass

    def encode_state(self, agent: Agent, goal: np.ndarray, enemies: List[EnemyAgent], obstacles_shapely: List[Polygon]) -> np.ndarray:
        goal_delta = goal - agent.pos
        dist_goal = np.linalg.norm(goal_delta)
        min_obs_dist = float('inf')
        for obs in obstacles_shapely:
            d = Point(agent.pos).distance(obs)
            if d < min_obs_dist:
                min_obs_dist = d
        sees_agent = 1.0 if any_enemy_sees(agent.pos, enemies, obstacles_shapely) else 0.0
        avg_enemy_dir = np.mean([e.direction for e in enemies], axis=0) if enemies else np.zeros(2)
        vec = np.array([
            agent.pos[0], agent.pos[1],
            agent.velocity[0], agent.velocity[1],
            goal_delta[0], goal_delta[1], dist_goal,
            min_obs_dist,
            sees_agent,
            avg_enemy_dir[0], avg_enemy_dir[1],
        ], dtype=np.float32)
        return vec

    def propose_action(self, agent: Agent, goal: np.ndarray, enemies: List[EnemyAgent], obstacles_shapely: List[Polygon], max_mag: float = 2.0) -> np.ndarray:
        v = self.encode_state(agent, goal, enemies, obstacles_shapely)
        self._ensure_model(len(v))
        with torch.no_grad():
            x = torch.from_numpy(np.tile(v, (self.k, 1))).float().to(self.device).unsqueeze(0)
            out = self.model(x)[:, -1, :]  # (1,2)
            a = out.squeeze(0).cpu().numpy()
        if np.linalg.norm(a) < 1e-6:
            # Fallback: move toward goal
            to_goal = goal - agent.pos
            if np.linalg.norm(to_goal) < 1e-6:
                return np.zeros(2)
            return normalize(to_goal) * max_mag
        return (a / (np.linalg.norm(a) + 1e-8)) * max_mag


def build_candidate_obstacles(obstacles: List[Tuple[Polygon, Tuple[float, float, float, float], np.ndarray]]) -> List[Dict]:
    candidates = []
    for idx, (poly, _bounds, centroid) in enumerate(obstacles):
        candidates.append({'index': idx, 'poly': poly, 'centroid': centroid})
    return candidates


def compute_action_mask(candidates: List[Dict], agent: Agent, goal: np.ndarray, min_dist_same: float = 2.0) -> np.ndarray:
    mask: List[bool] = []
    agent_to_goal = math.sqrt((agent.pos[0] - goal[0]) ** 2 + (agent.pos[1] - goal[1]) ** 2)
    for c in candidates:
        centroid = c['centroid']
        dist_agent_to_centroid = math.sqrt((agent.pos[0] - centroid[0]) ** 2 + (agent.pos[1] - centroid[1]) ** 2)
        if dist_agent_to_centroid < min_dist_same:
            mask.append(False)
            continue
        if agent_to_goal < dist_agent_to_centroid:
            mask.append(False)
            continue
        mask.append(True)
    return np.array(mask, dtype=bool)


# --------------------------
# Episode runners per method
# --------------------------
def run_episode(
    method: str,
    seed: int,
    cfg: SimulationConfig,
    dtqn_checkpoint: Optional[str] = None,
    record_trajectory: bool = False,
) -> EpisodeResult:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    steps = int(cfg.sim_time / cfg.dt)
    agent_start = np.array([cfg.scene_min, cfg.scene_min], dtype=float)
    agent_goal = np.array([cfg.scene_max, cfg.scene_max], dtype=float)
    agent = Agent(agent_start, agent_goal, max_speed=1.0)
    enemies = generate_enemies(cfg, rng)
    obstacles, obstacles_shapely = generate_obstacles(cfg, rng)
    add_goal_as_obstacle(obstacles, obstacles_shapely, agent_goal)
    # Prepare obstacles for fast point-in-polygon checks
    prepared_obstacles = [prep(obs) for obs in obstacles_shapely]

    result = EpisodeResult()
    last_pos = agent.pos.copy()

    # Trajectory recording
    if record_trajectory:
        result.agent_positions = [agent.pos.copy()]
        result.enemy_positions = [[e.pos.copy() for e in enemies]]
        result.enemy_directions = [[e.direction.copy() for e in enemies]]
        result.obstacles_shapely_snapshot = obstacles_shapely
        result.obstacles_snapshot = obstacles
        result.goal = agent_goal.copy()
        result.num_enemies = len(enemies)
        result.enemy_fovs = [e.fov for e in enemies]
        result.enemy_view_ranges = [e.view_range for e in enemies]

    # Method-specific state
    if method == 'dtqn_hl_ll':
        high_level = DTQNHighLevelPolicy(k=3, eps_greedy=0.1, checkpoint_path=dtqn_checkpoint)
    elif method == 'dtqn_hl_ll_nomem':
        # Memory-ablated DTQN: k=1 removes temporal aggregation
        high_level = DTQNHighLevelPolicy(k=1, eps_greedy=0.1, checkpoint_path=dtqn_checkpoint)
    elif method == 'dtqn_end2end':
        num_actions = 8
        end2end_model = DTQN(input_dim=10, output_dim=num_actions, k=3)  # Placeholder input_dim
        if dtqn_checkpoint and os.path.isfile(dtqn_checkpoint):
            try:
                end2end_model.load_state_dict(torch.load(dtqn_checkpoint, map_location='cpu'))
                end2end_model.eval()
            except Exception:
                pass
        action_dirs = [
            np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1]),
            np.array([1, 1]), np.array([-1, 1]), np.array([-1, -1]), np.array([1, -1]),
        ]

    # Rollout
    t = 0
    subgoal_centroid: Optional[np.ndarray] = None
    k_in_option = 0
    past_positions: List[np.ndarray] = []  # for stuck detection
    # Additional method-specific init
    if method == 'lstm_hl_mpc':
        lstm_high = LSTMHighLevelPolicy(k=3, eps_greedy=0.1, checkpoint_path=dtqn_checkpoint)
    elif method == 'lstm_mpc_end2end':
        lstm_end2end = LSTMEnd2EndWaypointPolicy(k=3, checkpoint_path=dtqn_checkpoint)
    elif method == 'lstm_end2end':
        lstm_action = LSTMEnd2EndActionPolicy(k=3, checkpoint_path=dtqn_checkpoint)
    elif method == 'encomp_cql':
        encomp_agent = EnCompCQLAgent(checkpoint_path=dtqn_checkpoint)
        encomp_cover = encomp_cover_map(obstacles_shapely, cfg.scene_min, cfg.scene_max)
        encomp_goal = encomp_goal_map(agent_goal, cfg.scene_min, cfg.scene_max)

    while t < steps:
        # Goal check
        goal_dist = float(np.linalg.norm(agent.pos - agent_goal))
        if goal_dist <= cfg.goal_reach_dist:
            result.success = True
            result.time_to_goal = t * cfg.dt
            break

        # Collision check
        # Use prepared geometries for speed
        p_pt = Point(agent.pos)
        if any(po.contains(p_pt) for po in prepared_obstacles):
            result.collision = True

        # Exposure accumulation (current-time FOVs)
        if enemies:
            if any(e.can_see(agent.pos, obstacles_shapely) for e in enemies):
                result.exposure_time += cfg.dt

        # Update enemies
        for enemy in enemies:
            enemy.update(cfg.dt, obstacles_shapely)

        # Control law per method
        if method in ('dtqn_hl_ll', 'dtqn_hl_ll_nomem'):
            # Early subgoal termination: re-select if reached subgoal or goal is close
            subgoal_reached = (subgoal_centroid is not None and np.linalg.norm(agent.pos - subgoal_centroid) < 2.0)
            # High-level select subgoal every option window or if none
            if subgoal_centroid is None or k_in_option >= cfg.max_k_per_option or subgoal_reached:
                candidates = build_candidate_obstacles(obstacles)
                mask = compute_action_mask(candidates, agent, agent_goal)
                if np.any(mask):
                    chosen_idx = high_level.select_subgoal(agent, enemies, candidates, mask, agent_goal, obstacles_shapely)
                    if chosen_idx == -1:
                        subgoal_centroid = agent_goal.copy()
                    else:
                        subgoal_centroid = candidates[chosen_idx]['centroid']
                else:
                    subgoal_centroid = agent_goal.copy()
                k_in_option = 0

            # Low-level forces to subgoal
            agent.goal = subgoal_centroid.copy()
            to_subgoal = agent.goal - agent.pos
            to_subgoal_norm = np.linalg.norm(to_subgoal)
            step_dir = (to_subgoal / to_subgoal_norm) if to_subgoal_norm > 1e-6 else np.zeros(2)
            probe_point = agent.pos + cfg.wait_probe_dist * step_dir
            probe_in_view = any_enemy_sees(probe_point, enemies, obstacles_shapely)
            min_enemy_dist = min(np.linalg.norm(agent.pos - e.pos) for e in enemies) if enemies else float('inf')
            proximity_risk = (min_enemy_dist < cfg.proximity_risk_dist)
            should_wait = probe_in_view or proximity_risk

            F_desired = (np.zeros(2) if should_wait else (desired_force(agent, desired_speed=cfg.desired_speed) * 10.0))
            F_obs = obstacle_force(agent, obstacles_shapely, threshold=cfg.obstacle_threshold, repulsive_coeff=cfg.obstacle_repulsive)
            F_enemy = np.zeros(2)
            F_anticipatory = np.zeros(2)
            for e in enemies:
                F_enemy += enemy_avoidance_force(agent, e, weight=cfg.enemy_avoid_weight, obstacles=obstacles_shapely)
                F_anticipatory += anticipatory_enemy_avoidance_force(agent, e, T=cfg.exposure_T_pred, weight=cfg.anticipate_weight)
            F_total = F_desired + F_obs + F_enemy + F_anticipatory
            # Stuck detection: add escape force if agent hasn't moved
            if len(past_positions) >= 100:
                recent = past_positions[-100:]
                total_dist = sum(np.linalg.norm(recent[i] - recent[i-1]) for i in range(1, len(recent)))
                if total_dist < 0.1:
                    # Find escape direction (max clearance away from recent average)
                    avg_pos = np.mean(past_positions[-5:], axis=0) if len(past_positions) >= 5 else agent.pos
                    avoid_dir = normalize(agent.pos - avg_pos) if np.linalg.norm(agent.pos - avg_pos) > 0.01 else np.array([1.0, 0.0])
                    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
                    best_clearance = -1.0
                    best_dir = normalize(agent_goal - agent.pos)
                    for ang in angles:
                        d = np.array([np.cos(ang), np.sin(ang)])
                        if np.dot(d, avoid_dir) < -0.7:
                            continue
                        ray = LineString([agent.pos, agent.pos + d * 10.0])
                        clr = 10.0  # default full clearance
                        for obs in obstacles_shapely:
                            inter = ray.intersection(obs)
                            if not inter.is_empty:
                                if inter.geom_type == 'Point':
                                    clr = min(clr, np.linalg.norm(np.array(inter.coords[0]) - agent.pos))
                                elif hasattr(inter, 'geoms'):
                                    for pt in inter.geoms:
                                        clr = min(clr, np.linalg.norm(np.array(pt.coords[0]) - agent.pos))
                        if clr > best_clearance:
                            best_clearance = clr
                            best_dir = d
                    F_total += best_dir * 25.0
            in_view = np.linalg.norm(F_enemy) > 0
            agent.update(F_total, cfg.dt, smoothing=0.9, max_speed=(1.0 if not in_view else 2.0))
            past_positions.append(agent.pos.copy())
            k_in_option += 1

        elif method == 'low_level_only':
            agent.goal = agent_goal.copy()
            # Potential fields with predicted FOV union penalty (as a force pushing away from union boundary)
            F_desired = desired_force(agent, desired_speed=cfg.desired_speed) * 10.0
            F_obs = obstacle_force(agent, obstacles_shapely, threshold=cfg.obstacle_threshold, repulsive_coeff=cfg.obstacle_repulsive)
            F_enemy = np.zeros(2)
            F_anticipatory = np.zeros(2)
            predicted_union = compute_predicted_fov_union(enemies, obstacles_shapely, T=cfg.exposure_T_pred)
            if not predicted_union.is_empty:
                p_pt = Point(agent.pos)
                # If inside predicted FoV union, push outward along gradient approximation
                if predicted_union.contains(p_pt):
                    nearest = predicted_union.boundary.interpolate(predicted_union.boundary.project(p_pt))
                    F_enemy += normalize(agent.pos - np.array([nearest.x, nearest.y])) * 500.0
            for e in enemies:
                F_anticipatory += anticipatory_enemy_avoidance_force(agent, e, T=cfg.exposure_T_pred, weight=cfg.anticipate_weight)
            F_total = F_desired + F_obs + F_enemy + F_anticipatory
            agent.update(F_total, cfg.dt, smoothing=0.9, max_speed=1.5)

        elif method == 'visibility_greedy':
            # Sample headings and pick the one minimizing exposure + distance-to-goal cost
            headings = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            best_score = float('inf')
            best_dir = np.zeros(2)
            for ang in headings:
                dvec = np.array([np.cos(ang), np.sin(ang)])
                candidate_pos = agent.pos + dvec * cfg.desired_speed * cfg.dt
                dist_goal = np.linalg.norm(agent_goal - candidate_pos)
                # Exposure proxy: is candidate in any current FoV
                exposure = 1.0 if any_enemy_sees(candidate_pos, enemies, obstacles_shapely) else 0.0
                # Obstacle clearance
                clearance_pen = 0.0
                for obs in obstacles_shapely:
                    dist = Point(candidate_pos).distance(obs)
                    if dist < cfg.obstacle_threshold:
                        clearance_pen += (cfg.obstacle_threshold - dist) * 50.0
                score = dist_goal + 5.0 * exposure + clearance_pen
                if score < best_score:
                    best_score = score
                    best_dir = dvec
            F = best_dir * 10.0
            agent.update(F, cfg.dt, smoothing=0.5, max_speed=1.0)

        elif method == 'dwa_fov':
            # Simple DWA: sample (v, w); roll short horizon; evaluate cost
            v_samples = np.linspace(0.0, 2.0, 5)
            w_samples = np.linspace(-1.5, 1.5, 7)
            horizon = 1.0
            steps_h = max(1, int(horizon / cfg.dt))
            best_cost = float('inf')
            best_v, best_w = 0.0, 0.0
            heading = normalize(agent.velocity) if np.linalg.norm(agent.velocity) > 1e-3 else normalize(agent.goal - agent.pos)
            for v in v_samples:
                for w in w_samples:
                    pos = agent.pos.copy()
                    theta = math.atan2(heading[1], heading[0])
                    cost = 0.0
                    for _ in range(steps_h):
                        theta += w * cfg.dt
                        pos += np.array([np.cos(theta), np.sin(theta)]) * v * cfg.dt
                        dist_goal = np.linalg.norm(agent_goal - pos)
                        obs_pen = 0.0
                        for obs in obstacles_shapely:
                            d = Point(pos).distance(obs)
                            if d < cfg.obstacle_threshold:
                                obs_pen += (cfg.obstacle_threshold - d) * 200.0
                        exp_pen = 5.0 if any_enemy_sees(pos, enemies, obstacles_shapely) else 0.0
                        cost += dist_goal + obs_pen + exp_pen
                    if cost < best_cost:
                        best_cost = cost
                        best_v, best_w = v, w
            theta0 = math.atan2(heading[1], heading[0]) + best_w * cfg.dt
            move_dir = np.array([np.cos(theta0), np.sin(theta0)])
            agent.update(move_dir * best_v * 2.0, cfg.dt, smoothing=0.5, max_speed=2.0)

        elif method == 'vfh_plus_fov':
            # Basic VFH+: build histogram; find valley towards goal while avoiding FOV sectors
            num_bins = 36
            bin_angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
            histogram = np.zeros(num_bins)
            # Obstacle contributions
            for obs in obstacles_shapely:
                # Sample points along boundary
                coords = np.array(obs.exterior.coords)
                for p in coords[:: max(1, len(coords) // 20)]:
                    vec = np.array(p) - agent.pos
                    d = np.linalg.norm(vec)
                    if d < 1e-6:
                        continue
                    ang = (math.atan2(vec[1], vec[0]) + 2 * np.pi) % (2 * np.pi)
                    bin_idx = int((ang / (2 * np.pi)) * num_bins) % num_bins
                    histogram[bin_idx] += 1.0 / (d * d)
            # FoV penalty
            for enemy in enemies:
                vis = enemy.get_visibility_polygon(obstacles_shapely)
                # Penalize bins that point into current FoV region around agent
                for bi, ang in enumerate(bin_angles):
                    look = agent.pos + np.array([np.cos(ang), np.sin(ang)]) * 1.0
                    if vis.contains(Point(look)):
                        histogram[bi] += 5.0
            # Choose lowest histogram bin near goal direction
            to_goal = agent.goal - agent.pos
            goal_ang = (math.atan2(to_goal[1], to_goal[0]) + 2 * np.pi) % (2 * np.pi)
            best_bin = None
            best_score = float('inf')
            for bi, ang in enumerate(bin_angles):
                # Combine histogram magnitude and deviation from goal
                dev = min(abs(ang - goal_ang), 2 * np.pi - abs(ang - goal_ang))
                score = histogram[bi] + 0.5 * dev
                if score < best_score:
                    best_score = score
                    best_bin = bi
            chosen_ang = bin_angles[best_bin] if best_bin is not None else goal_ang
            move_dir = np.array([np.cos(chosen_ang), np.sin(chosen_ang)])
            agent.update(move_dir * 2.0, cfg.dt, smoothing=0.7, max_speed=1.5)

        elif method == 'end2end_rl':
            # Simple heuristic stub for now: greedy with obstacle+exposure penalties (can be replaced by PPO model)
            headings = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            best_score = float('inf')
            best_dir = np.zeros(2)
            for ang in headings:
                dvec = np.array([np.cos(ang), np.sin(ang)])
                candidate_pos = agent.pos + dvec * cfg.desired_speed * cfg.dt
                dist_goal = np.linalg.norm(agent_goal - candidate_pos)
                exposure = 1.0 if any_enemy_sees(candidate_pos, enemies, obstacles_shapely) else 0.0
                obs_pen = 0.0
                for obs in obstacles_shapely:
                    dist = Point(candidate_pos).distance(obs)
                    if dist < cfg.obstacle_threshold:
                        obs_pen += (cfg.obstacle_threshold - dist) * 200.0
                score = dist_goal + 10.0 * exposure + obs_pen
                if score < best_score:
                    best_score = score
                    best_dir = dvec
            agent.update(best_dir * 2.0, cfg.dt, smoothing=0.5, max_speed=1.5)

        elif method == 'lstm_hl_mpc':
            # High-level LSTM subgoal selection at option boundaries
            if subgoal_centroid is None or k_in_option >= cfg.max_k_per_option:
                candidates = build_candidate_obstacles(obstacles)
                mask = compute_action_mask(candidates, agent, agent_goal)
                if np.any(mask):
                    chosen_idx = lstm_high.select_subgoal(agent, enemies, candidates, mask, agent_goal, obstacles_shapely)
                    if chosen_idx == -1:
                        subgoal_centroid = agent_goal.copy()
                    else:
                        subgoal_centroid = candidates[chosen_idx]['centroid']
                else:
                    subgoal_centroid = agent_goal.copy()
                k_in_option = 0
            # MPC to subgoal
            u = mpc_control_step(agent=agent, target=subgoal_centroid, enemies=enemies, obstacles_shapely=obstacles_shapely, cfg=cfg)
            agent.update(u, cfg.dt, smoothing=0.9, max_speed=cfg.mpc_speed_limit)
            k_in_option += 1

        elif method == 'lstm_mpc_end2end':
            # LSTM proposes a local waypoint; MPC moves toward it
            local_wp = lstm_end2end.propose_local_waypoint(agent, agent_goal, enemies, obstacles_shapely, max_radius=3.0)
            u = mpc_control_step(agent=agent, target=local_wp, enemies=enemies, obstacles_shapely=obstacles_shapely, cfg=cfg)
            agent.update(u, cfg.dt, smoothing=0.9, max_speed=cfg.mpc_speed_limit)

        elif method == 'lstm_end2end':
            # LSTM outputs direct action; environment dynamics apply smoothing and speed cap
            a = lstm_action.propose_action(agent, agent_goal, enemies, obstacles_shapely, max_mag=2.0)
            agent.update(a, cfg.dt, smoothing=0.9, max_speed=2.0)

        elif method == 'encomp_cql':
            maps_enc, pos_enc = encomp_generate_maps(
                agent.pos, agent_goal, enemies, obstacles_shapely,
                cfg.scene_min, cfg.scene_max,
                precomputed_cover=encomp_cover, precomputed_goal=encomp_goal,
            )
            act_idx = encomp_agent.select_action(maps_enc, pos_enc, epsilon=0.0)
            force = encomp_action_to_force(act_idx, speed=2.0)
            # Obstacle avoidance safety layer (same as other baselines)
            F_obs = obstacle_force(agent, obstacles_shapely, threshold=cfg.obstacle_threshold, repulsive_coeff=cfg.obstacle_repulsive)
            agent.update(force + F_obs, cfg.dt, smoothing=0.7, max_speed=2.0)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Enemy updates already applied; accumulate metrics
        step_len = float(np.linalg.norm(agent.pos - last_pos))
        result.path_length += step_len
        last_pos = agent.pos.copy()
        result.steps_taken += 1
        t += 1

        # Record trajectory for video
        if record_trajectory:
            result.agent_positions.append(agent.pos.copy())
            result.enemy_positions.append([e.pos.copy() for e in enemies])
            result.enemy_directions.append([e.direction.copy() for e in enemies])

    return result


# --------------------------
# Batch evaluation
# --------------------------
def run_evaluation(method: str, num_runs: int, cfg: SimulationConfig, dtqn_checkpoint: Optional[str] = None, base_seed: int = 0,
                   video_dir: Optional[str] = None, video_episodes: Optional[List[int]] = None) -> List[EpisodeResult]:
    """Run batch evaluation. Optionally save videos for selected episode indices.
    
    Args:
        video_dir: Directory to save videos. If None, no videos saved.
        video_episodes: List of episode indices (0-based) to record videos for.
                        If None but video_dir is set, records first 5 episodes.
    """
    if video_dir and video_episodes is None:
        video_episodes = list(range(min(5, num_runs)))
    video_episodes = set(video_episodes or [])
    results: List[EpisodeResult] = []
    for i in range(num_runs):
        seed = base_seed + i
        record = (i in video_episodes) and video_dir is not None
        res = run_episode(method=method, seed=seed, cfg=cfg, dtqn_checkpoint=dtqn_checkpoint, record_trajectory=record)
        results.append(res)
        tag = 'OK' if res.success else 'FAIL'
        print(f"  [{i+1:3d}/{num_runs}] seed={seed}  {tag}  steps={res.steps_taken}  path={res.path_length:.1f}  exposure={res.exposure_time:.1f}")
        if record and video_dir:
            outcome = 'success' if res.success else 'fail'
            out_path = os.path.join(video_dir, f"{method}_ep{i:03d}_seed{seed}_{outcome}.mp4")
            try:
                save_episode_video(res, out_path, cfg=cfg, title=f"{method} ep{i} seed{seed} [{outcome.upper()}]")
            except Exception as e:
                print(f"  Warning: video save failed for ep{i}: {e}")
    return results


def summarize_results(results: List[EpisodeResult]) -> Dict[str, float]:
    success_rate = np.mean([1.0 if r.success else 0.0 for r in results]) if results else 0.0
    collision_rate = np.mean([1.0 if r.collision else 0.0 for r in results]) if results else 0.0
    path_lengths = np.array([r.path_length for r in results], dtype=float)
    times = np.array([r.time_to_goal for r in results], dtype=float)
    exposures = np.array([r.exposure_time for r in results], dtype=float)
    # Treat non-success times as NaN for central tendency of time-to-goal
    times[~np.isfinite(times)] = np.nan
    summary = {
        'success_rate': float(success_rate),
        'collision_rate': float(collision_rate),
        'path_length_mean': float(np.nanmean(path_lengths)),
        'path_length_median': float(np.nanmedian(path_lengths)),
        'path_length_std': float(np.nanstd(path_lengths)),
        'path_length_min': float(np.nanmin(path_lengths)),
        'path_length_max': float(np.nanmax(path_lengths)),
        'time_to_goal_mean': float(np.nanmean(times)),
        'time_to_goal_median': float(np.nanmedian(times)),
        'time_to_goal_std': float(np.nanstd(times)),
        'time_to_goal_min': float(np.nanmin(times)),
        'time_to_goal_max': float(np.nanmax(times)),
        'exposure_time_mean': float(np.nanmean(exposures)),
        'exposure_time_median': float(np.nanmedian(exposures)),
        'exposure_time_std': float(np.nanstd(exposures)),
        'exposure_time_min': float(np.nanmin(exposures)),
        'exposure_time_max': float(np.nanmax(exposures)),
    }
    return summary


def save_results_csv(results: List[EpisodeResult], summary: Dict[str, float], out_path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['run_idx', 'success', 'steps', 'path_length', 'time_to_goal', 'collision', 'exposure_time'])
        for i, r in enumerate(results):
            writer.writerow([i, int(r.success), r.steps_taken, f"{r.path_length:.6f}", ("" if not r.success else f"{r.time_to_goal:.6f}"), int(r.collision), f"{r.exposure_time:.6f}"])
        writer.writerow([])
        writer.writerow(['metric', 'value'])
        for k, v in summary.items():
            writer.writerow([k, v])


