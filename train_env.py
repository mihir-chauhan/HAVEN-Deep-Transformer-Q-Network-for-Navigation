import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib import animation
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from IPython.display import Video
import copy
import threading
import math
import os
import csv
import atexit
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

# Shapely is used for geometric computations (polygons, intersections, etc.)
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union
from dtqn_model import DTQN, PrioritizedReplayBuffer, ObservationHistory, RunningNormalizer

# Video writer detection (prefer ffmpeg, fallback to pillow)
try:
    from matplotlib.animation import FFMpegWriter
    VIDEO_WRITER = 'ffmpeg'
except (ImportError, RuntimeError):
    try:
        from matplotlib.animation import PillowWriter
        VIDEO_WRITER = 'pillow'
    except (ImportError, RuntimeError):
        VIDEO_WRITER = None

# --------------------------
# Scene Setup & Utility Functions
# --------------------------

# Define the simulation boundaries.
scene_min, scene_max = -5, 15

# --------------------------
# Hierarchical Controller Config
# --------------------------
USE_DTQN = True           # If True, use DTQNPolicy; otherwise use HeuristicPolicy
TRAIN_HIGH_LEVEL = True   # Enable online updates for high-level DTQN
GAMMA = 0.99
MAX_K_PER_OPTION = 20     # Low-level steps executed per chosen subgoal before re-evaluating
EPS_GREEDY = 0.1          # Exploration for DTQN policy
DTQN_K = 3                # Sequence length for DTQN inputs
DTQN_LR = 1e-3
DTQN_SAVE_INTERVAL = 50   # Save model every N updates

# Paths
_BASE_DIR = os.path.dirname(__file__)
DTQN_CHECKPOINT_PATH = os.path.join(_BASE_DIR, 'dtqn_checkpoint.pt')
LOG_CSV_PATH = os.path.join(_BASE_DIR, 'high_level_log.csv')
TRAIN_METRICS_SUBDIR = os.environ.get('VIDEO_SUBDIR', 'train')
TRAIN_METRICS_PATH = os.path.join(_BASE_DIR, TRAIN_METRICS_SUBDIR, 'training_metrics.csv')

def _get_bool_env(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    val = str(val).strip().lower()
    return val in ('1', 'true', 't', 'yes', 'y')

# Env overrides for external runners
USE_DTQN = _get_bool_env('USE_DTQN', USE_DTQN)
TRAIN_HIGH_LEVEL = _get_bool_env('TRAIN_HIGH_LEVEL', TRAIN_HIGH_LEVEL)
DISABLE_VIDEO = _get_bool_env('DISABLE_VIDEO', False)

REWARD_WEIGHTS = {
    'progress_to_subgoal': 1.0,
    'goal_progress': 0.5,
    'exposure_penalty': 2.0,
    'collision_penalty': 5.0,
    'time_penalty': 0.10,
    'subgoal_bonus': 3.0,
    'goal_bonus': 200.0,
    'failure_penalty': 200.0,
}

# Training stability improvements
EPSILON_DECAY_SCHEDULE = 'gradual'  # 'fast' or 'gradual'

def normalize(v):
    """
    Returns the unit vector in the direction of vector v.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def is_agent_stuck(agent, past_positions, threshold=0.1, window=100):
    """
    Determines if an agent is stuck by analyzing its recent movement history.
    
    Parameters:
        agent: The agent to check
        past_positions: List of past positions (most recent positions at the end)
        threshold: Maximum distance considered as "not moving"
        window: Number of past positions to consider
    
    Returns:
        Boolean indicating if the agent is stuck
    """
    if len(past_positions) < window + 1:
        return False
        
    # Get the last 'window' positions
    recent_positions = past_positions[-window:]
    
    # Calculate the total distance moved over the last 'window' positions
    total_distance = 0
    for i in range(1, len(recent_positions)):
        total_distance += np.linalg.norm(recent_positions[i] - recent_positions[i-1])
    
    # If the agent hasn't moved much relative to its max speed, it might be stuck
    return total_distance < threshold
def find_escape_direction(agent, obstacles_shapely, past_positions, num_samples=16):
    """
    Finds a direction to escape when the agent is stuck.
    
    Parameters:
        agent: The agent that is stuck
        obstacles_shapely: List of obstacle polygons
        past_positions: Recent history of agent positions
        num_samples: Number of directions to sample
    
    Returns:
        Best direction to escape as a unit vector
    """
    # Find the average position over the last few steps
    if len(past_positions) < 5:
        avg_pos = agent.pos
    else:
        avg_pos = np.mean(past_positions[-5:], axis=0)
    
    # Direction from average position to current position (to avoid going back)
    avoid_dir = normalize(agent.pos - avg_pos) if np.linalg.norm(agent.pos - avg_pos) > 0.01 else np.array([1, 0])
    
    # Sample directions evenly around a circle
    angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
    directions = []
    clearances = []
    
    for angle in angles:
        # Generate a direction vector for this angle
        direction = np.array([np.cos(angle), np.sin(angle)])
        
        # Skip directions that would lead back to where we came from
        if np.dot(direction, avoid_dir) < -0.7:  # Avoid going directly back
            continue
            
        # Check how far we can go in this direction before hitting an obstacle
        clearance = float('inf')
        ray = LineString([agent.pos, agent.pos + direction * 10])  # Create a long ray
        
        for obs in obstacles_shapely:
            inter = ray.intersection(obs)
            if not inter.is_empty:
                if inter.geom_type == 'Point':
                    dist = np.linalg.norm(np.array(inter.coords[0]) - agent.pos)
                    clearance = min(clearance, dist)
                elif inter.geom_type == 'MultiPoint':
                    for pt in inter:
                        dist = np.linalg.norm(np.array(pt.coords[0]) - agent.pos)
                        clearance = min(clearance, dist)
                elif inter.geom_type == 'LineString':
                    dist = np.linalg.norm(np.array(inter.coords[0]) - agent.pos)
                    clearance = min(clearance, dist)
        
        directions.append(direction)
        clearances.append(clearance)
    
    if not directions:  # If all directions were rejected, include all
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            directions.append(direction)
            
            clearance = float('inf')
            ray = LineString([agent.pos, agent.pos + direction * 10])
            
            for obs in obstacles_shapely:
                inter = ray.intersection(obs)
                if not inter.is_empty:
                    if inter.geom_type == 'Point':
                        dist = np.linalg.norm(np.array(inter.coords[0]) - agent.pos)
                        clearance = min(clearance, dist)
                    elif inter.geom_type == 'MultiPoint':
                        for pt in inter:
                            dist = np.linalg.norm(np.array(pt.coords[0]) - agent.pos)
                            clearance = min(clearance, dist)
                    elif inter.geom_type == 'LineString':
                        dist = np.linalg.norm(np.array(inter.coords[0]) - agent.pos)
                        clearance = min(clearance, dist)
            
            clearances.append(clearance)
    
    # Weight directions by clearance and dot product with goal direction
    if not clearances:
        return normalize(agent.goal - agent.pos)
    
    # Find index of maximum clearance
    best_idx = np.argmax(clearances)
    return directions[best_idx]

def improved_obstacle_force(entity, obstacles, threshold=1.5, repulsive_coeff=150.0):
    """
    Enhanced version of obstacle force function that increases repulsion
    when very close to obstacles.
    """
    force = np.zeros(2)
    entity_point = Point(entity.pos)
    
    for obs in obstacles:
        # Find the nearest point on the obstacle to the entity
        if entity_point.distance(obs) < threshold:
            # If we're close to the obstacle, find the closest point on its boundary
            nearest_points = nearest_points_on_polygon(entity.pos, obs)
            if nearest_points:
                nearest_point = np.array(nearest_points[0])
                d = np.linalg.norm(entity.pos - nearest_point)
                
                # Exponential force scaling for very close distances
                if d < 0.5:  # Really close to obstacle
                    scaling = repulsive_coeff * (3.0 - 1.0/threshold) * (1.0/d)
                else:
                    scaling = repulsive_coeff * (1.0/d - 1.0/threshold)
                
                force += scaling * normalize(entity.pos - nearest_point)
    return force


# --------------------------
# Visibility Polygon via Ray-Casting
# --------------------------
def compute_visibility_polygon_raycast(pos, direction, fov, view_range, obstacles, num_rays=50, secondary_view_range_factor=0.3):
    """
    (Patched) Computes the field-of-view polygon for an enemy using ray-casting with robust
    safeguards against degenerate polygons.
    """
    points = []
    secondary_range = view_range * secondary_view_range_factor
    primary_rays = num_rays
    rel_angles_primary = np.linspace(-fov/2, fov/2, primary_rays)
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
                        nearest_dist = d; nearest_point = candidate
                elif inter.geom_type == 'MultiPoint':
                    for pt in inter:
                        candidate = np.array(pt.coords[0]); d = np.linalg.norm(candidate - pos)
                        if d < nearest_dist:
                            nearest_dist = d; nearest_point = candidate
                elif inter.geom_type == 'LineString':
                    candidate = np.array(inter.coords[0]); d = np.linalg.norm(candidate - pos)
                    if d < nearest_dist:
                        nearest_dist = d; nearest_point = candidate
        points.append(nearest_point)
    rel_angles_secondary = np.linspace(fov/2, 2*np.pi - fov/2, num_rays)
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
                    candidate = np.array(inter.coords[0]); d = np.linalg.norm(candidate - pos)
                    if d < nearest_dist:
                        nearest_dist = d; nearest_point = candidate
                elif inter.geom_type == 'MultiPoint':
                    for pt in inter:
                        candidate = np.array(pt.coords[0]); d = np.linalg.norm(candidate - pos)
                        if d < nearest_dist:
                            nearest_dist = d; nearest_point = candidate
                elif inter.geom_type == 'LineString':
                    candidate = np.array(inter.coords[0]); d = np.linalg.norm(candidate - pos)
                    if d < nearest_dist:
                        nearest_dist = d; nearest_point = candidate
        points.append(nearest_point)

    poly_coords = [pos] + points

    # Deduplicate with rounding to avoid almost-identical points causing invalid rings
    unique = []
    seen = set()
    for c in poly_coords:
        x, y = float(c[0]), float(c[1])
        key = (round(x, 6), round(y, 6))
        if key not in seen:
            seen.add(key)
            unique.append([x, y])
    if len(unique) < 3:
        # Fallback small square around pos
        r = 0.1
        sq = [[pos[0]+r, pos[1]+r], [pos[0]-r, pos[1]+r], [pos[0]-r, pos[1]-r], [pos[0]+r, pos[1]-r], [pos[0]+r, pos[1]+r]]
        return Polygon(sq)
    # Ensure closed (repeat first point)
    if unique[0] != unique[-1]:
        unique.append(unique[0])
    try:
        poly = Polygon(unique)
        if (not poly.is_valid) or poly.area <= 1e-9:
            poly = poly.buffer(0)  # attempt fix self-intersections
        if (not poly.is_valid) or poly.area <= 1e-9:
            r = 0.1
            sq = [[pos[0]+r, pos[1]+r], [pos[0]-r, pos[1]+r], [pos[0]-r, pos[1]-r], [pos[0]+r, pos[1]-r], [pos[0]+r, pos[1]+r]]
            poly = Polygon(sq)
        return poly
    except Exception:
        r = 0.1
        sq = [[pos[0]+r, pos[1]+r], [pos[0]-r, pos[1]+r], [pos[0]-r, pos[1]-r], [pos[0]+r, pos[1]-r], [pos[0]+r, pos[1]+r]]
        return Polygon(sq)

# --------------------------
# Advanced Intent Prediction Model (Logistic Mixture)
# --------------------------
def predict_detection_probability(point, enemy, T=10.0, angle_threshold=0.3, dist_threshold=2.0, a=10.0, b=10.0):
    """
    Predicts the probability that the enemy will detect the agent at a given point T seconds in the future.
    Uses a logistic mixture model based on the angular difference and distance from the predicted enemy position.

    Parameters:
        point          : The point (agent position) to evaluate.
        enemy          : The enemy object.
        T              : Prediction horizon in seconds.
        angle_threshold: Threshold angle (in radians) for high detection risk.
        dist_threshold : Threshold distance for high detection risk.
        a, b           : Scaling factors for the logistic functions.

    Returns:
        A probability (0 to 1) of detection.
    """
    # Predict the enemy's future position.
    predicted_pos = enemy.pos + enemy.direction * enemy.speed * T
    v = point - predicted_pos
    d = np.linalg.norm(v)
    if d == 0:
        return 1.0
    v_norm = v / d
    # Compute the angular difference between the ray from predicted position to point and enemy's direction.
    angle_diff = np.arccos(np.clip(np.dot(v_norm, enemy.direction), -1.0, 1.0))
    # Logistic functions for angle and distance.
    p_angle = 1.0 / (1.0 + np.exp(a * (angle_diff - angle_threshold)))
    p_dist  = 1.0 / (1.0 + np.exp(b * (d - dist_threshold)))
    return p_angle * p_dist

# --------------------------
# Force Functions for Main Agent
# --------------------------
def desired_force(agent, desired_speed=1.0):
    """
    Computes a force that pulls the agent toward its goal.
    """
    diff = agent.goal - agent.pos
    return desired_speed * normalize(diff)

def obstacle_force(entity, obstacles, threshold=1.0, repulsive_coeff=100.0):
    """
    Computes a repulsive force from obstacles if the entity (agent or enemy) is too close.
    Modified to work with convex polygons. Hardened to avoid division by zero / NaNs.
    """
    force = np.zeros(2)
    entity_point = Point(entity.pos)
    eps = 1e-8
    for obs in obstacles:
        if entity_point.distance(obs) < threshold:
            nearest_points = nearest_points_on_polygon(entity.pos, obs)
            if nearest_points:
                nearest_point = np.array(nearest_points[0])
                d = np.linalg.norm(entity.pos - nearest_point)
                if d < threshold and d > eps:
                    coeff = (1.0/d - 1.0/threshold)
                    if np.isfinite(coeff):
                        force += repulsive_coeff * coeff * normalize(entity.pos - nearest_point)
    return force

def nearest_points_on_polygon(point, polygon):
    """
    Find the nearest point on a polygon to a given point.
    """
    # Convert point to a Shapely Point
    point_geom = Point(point)
    
    # If the point is inside the polygon, return the point itself
    if polygon.contains(point_geom):
        return [(point[0], point[1]), (point[0], point[1])]
    
    # Get the boundary of the polygon
    boundary = polygon.boundary
    
    # Find the nearest point on the boundary
    nearest_point = boundary.interpolate(boundary.project(point_geom))
    
    return [(nearest_point.x, nearest_point.y), (point[0], point[1])]

def fast_escape_force(agent, enemies, obstacles, weight=500.0):
    escape_force = np.zeros(2)
    for enemy in enemies:
        if enemy.can_see(agent.pos, obstacles):
            diff = agent.pos - enemy.pos
            d = np.linalg.norm(diff)
            if d == 0:
                continue
            escape_direction = normalize(diff)
            escape_intensity = weight / (d + 0.1)  # Stronger force when closer
            escape_force += escape_intensity * escape_direction
    return escape_force

def fast_escape_force_dir(agent, enemy, T=10.0, weight=1500.0, epsilon=0.1):
    predicted_pos = enemy.pos + enemy.direction * enemy.speed * T
    v = agent.pos - predicted_pos
    d = np.linalg.norm(v)
    if d == 0:
        return np.zeros(2)

    p = predict_detection_probability(agent.pos, enemy, T=T)

    # Determine the shortest escape route from the enemy's FOV
    fov_direction = enemy.direction
    to_agent = normalize(v)
    escape_direction = to_agent - np.dot(to_agent, fov_direction) * fov_direction
    escape_direction = normalize(escape_direction) if np.linalg.norm(escape_direction) > 0 else to_agent

    return escape_direction * (weight * (p**5) / (d + epsilon))

def enemy_avoidance_force(agent, enemy, weight=50.0, obstacles=None, epsilon=0.1):
    """
    Computes a repulsive force if the agent is currently in an enemy's field of view.
    """
    if enemy.can_see(agent.pos, obstacles):
        diff = agent.pos - enemy.pos
        d = np.linalg.norm(diff)
        if d == 0:
            return np.zeros(2)
        return normalize(diff) * (weight / ((d+epsilon)**2))
    return np.zeros(2)

def anticipatory_enemy_avoidance_force(agent, enemy, T=10.0, weight=1500.0, epsilon=0.1):
    """
    Computes a repulsive force based on the enemy's predicted position T seconds ahead.
    Uses a nonlinear scaling (p**5) to strongly repel the agent if detection risk is high.
    """
    predicted_pos = enemy.pos + enemy.direction * enemy.speed * T
    v = agent.pos - predicted_pos
    d = np.linalg.norm(v)
    if d == 0:
        return np.zeros(2)
    p = predict_detection_probability(agent.pos, enemy, T=T)
    return normalize(v) * (weight * (p**5) / (d+epsilon))

# --------------------------
# Agent and Enemy Classes
# --------------------------
class Agent:
    def __init__(self, pos, goal, max_speed=1.0):
        self.pos = np.array(pos, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.velocity = np.zeros(2)
        self.max_speed = max_speed

    def update(self, force, dt, smoothing=0.9, max_speed=1.0):
        """
        Updates the agent's velocity and position based on the applied force.
        Exponential smoothing is used to ensure smooth motion.
        """
        self.velocity = smoothing * self.velocity + (1-smoothing) * force
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity/speed) * max_speed
        self.pos += self.velocity * dt

class EnemyAgent:
    # If you want to add a parameter to control the secondary FOV range factor:
    def __init__(self, pos, direction, fov=np.pi/3, view_range=5.0, speed=0.3, secondary_range_factor=0.3):
        self.pos = np.array(pos, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))
        self.fov = fov
        self.view_range = view_range
        self.speed = speed
        self.secondary_range_factor = secondary_range_factor
        self._cached_vis_poly = None
        self._cache_key = None

    def update(self, dt, obstacles):
        """
        Updates the enemy's position and direction.
        The enemy applies an obstacle repulsive force to avoid static obstacles,
        bounces off scene boundaries, and applies a slight random rotation.
        """
        F_obs = obstacle_force(self, obstacles, threshold=0.5, repulsive_coeff=200.0)
        self.direction = normalize(self.direction + 0.3 * F_obs)
        self.pos += self.direction * self.speed * dt
        # Bounce off scene boundaries.
        if self.pos[0] < scene_min or self.pos[0] > scene_max:
            self.direction[0] = -self.direction[0]
        if self.pos[1] < scene_min or self.pos[1] > scene_max:
            self.direction[1] = -self.direction[1]
        # Apply a slight random rotation.
        angle = 0.01 * dt
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        self.direction = normalize(np.dot(rot, self.direction))
        # Invalidate visibility cache after movement
        self._cached_vis_poly = None
        self._cache_key = None

    def can_see(self, point, obstacles):
        """
        Fast single-ray visibility check.
        Instead of building the full 100-ray polygon and doing point-in-polygon,
        we: (1) check range, (2) check angular cone, (3) cast ONE ray to the target
        and see if obstacles block it.  ~100x faster than the polygon approach.
        """
        diff = np.asarray(point) - self.pos
        dist = np.linalg.norm(diff)
        if dist < 1e-9:
            return True  # enemy is on top of the point

        # Determine which zone the point is in and the effective range
        cos_angle = np.dot(self.direction, diff / dist)  # cosine of angle between direction and diff
        half_fov_cos = np.cos(self.fov / 2)

        if cos_angle >= half_fov_cos:
            # Inside primary FOV cone
            effective_range = self.view_range
        else:
            # Outside primary FOV → secondary (rear awareness) zone
            effective_range = self.view_range * self.secondary_range_factor

        if dist > effective_range:
            return False  # too far away

        # Cast a single ray from enemy to the target point and check for blocking obstacles
        ray = LineString([self.pos, point])
        for obs in obstacles:
            inter = ray.intersection(obs)
            if inter.is_empty:
                continue
            # Find closest intersection distance
            if inter.geom_type == 'Point':
                d = np.linalg.norm(np.array(inter.coords[0]) - self.pos)
            elif inter.geom_type == 'MultiPoint':
                d = min(np.linalg.norm(np.array(pt.coords[0]) - self.pos) for pt in inter.geoms)
            elif inter.geom_type == 'LineString':
                d = np.linalg.norm(np.array(inter.coords[0]) - self.pos)
            else:
                d = 0.0
            if d < dist - 1e-6:
                return False  # obstacle blocks the line of sight
        return True

        
    def get_visibility_polygon(self, obstacles):
        """
        Computes the enemy's current field of view polygon.
        Uses per-step caching: the expensive raycast is only done once per
        position/direction, then reused for all can_see() calls in the same step.
        """
        key = (round(self.pos[0], 8), round(self.pos[1], 8),
               round(self.direction[0], 8), round(self.direction[1], 8))
        if self._cached_vis_poly is not None and self._cache_key == key:
            return self._cached_vis_poly
        poly = compute_visibility_polygon_raycast(
            self.pos, 
            self.direction, 
            self.fov, 
            self.view_range, 
            obstacles,
            num_rays=100,
            secondary_view_range_factor=self.secondary_range_factor
        )
        self._cached_vis_poly = poly
        self._cache_key = key
        return poly

# Function to create random convex polygons
def create_random_convex_polygon(sides=5):
    """
    Create a random convex polygon with at most 'sides' sides.
    """
    # Generate random points
    sides = np.random.randint(3, sides + 1)  # Between 3 and max sides
    center_x = np.random.uniform(scene_min, scene_max - 3)
    center_y = np.random.uniform(scene_min, scene_max - 3)
    
    # Generate points in a circle around the center
    radius = np.random.uniform(0.5, 2.0)
    angles = np.sort(np.random.uniform(0, 2*np.pi, sides))
    
    # Create coordinates
    coords = []
    for angle in angles:
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        coords.append((x, y))
    
    # Create a Shapely polygon
    poly = Polygon(coords)
    return poly

# Function to get centroid of a polygon
def get_polygon_centroid(polygon):
    """
    Returns the centroid coordinates of a polygon.
    """
    return np.array([polygon.centroid.x, polygon.centroid.y])

# --------------------------
# Hierarchical Policy and Helpers
# --------------------------
class HighLevelPolicy:
    def select_subgoal(self, state, candidates, mask):
        """
        Return index into candidates list. If none valid, return -1.
        state: dict of features (not used by heuristic baseline)
        candidates: List[dict] with keys: index, centroid, poly
        mask: np.ndarray[bool] same length as candidates
        """
        raise NotImplementedError

class HeuristicPolicy(HighLevelPolicy):
    def select_subgoal(self, state, candidates, mask):
        # Choose the valid candidate whose centroid is closest to the final goal
        if len(candidates) == 0:
            return -1
        valid_indices = [i for i, m in enumerate(mask) if m]
        if not valid_indices:
            return -1
        goal = state['goal']
        best_i = -1
        best_score = float('inf')
        for i in valid_indices:
            c = candidates[i]['centroid']
            # score: distance from candidate centroid to final goal (smaller is better)
            dx = float(c[0]) - goal[0]
            dy = float(c[1]) - goal[1]
            score = dx*dx + dy*dy
            if score < best_score:
                best_score = score
                best_i = i
        return best_i

class DTQNPolicy(HighLevelPolicy):
    def __init__(self, device=None, train_mode=True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = DTQN_K
        self.input_dim = None
        self.model = None
        self.target_model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.SmoothL1Loss(reduction='none')
        self.eps = 1.0 if train_mode else 0.05
        self.eps_min = 0.15 if train_mode else 0.05
        self.eps_decay = 0.9998
        self.update_count = 0
        self.tau = 0.005
        self.batch_size = 64
        self.min_buffer_size = 200
        self.train_mode = train_mode
        self.grad_clip = 5.0
        self.train_every = 4
        self.decision_count = 0

        self.replay_buffer = PrioritizedReplayBuffer(capacity=50000)
        self.obs_history = ObservationHistory(k=self.k, dim=16)
        self.normalizer = RunningNormalizer(dim=16)

        self.last_q_values = []
        self.last_selected = None
        self.last_selected_q = None
        self.last_chosen_seq = None
        self.checkpoint_path = DTQN_CHECKPOINT_PATH

    def _encode_raw(self, agent, enemies, candidate, goal, obstacles_shapely):
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

        return np.array([
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

    def encode_state_candidate(self, agent, enemies, candidate, goal, obstacles_shapely):
        raw = self._encode_raw(agent, enemies, candidate, goal, obstacles_shapely)
        if self.train_mode:
            self.normalizer.update(raw)
        return self.normalizer.normalize(raw)

    def _ensure_model(self, input_dim):
        if self.model is None:
            self.input_dim = input_dim
            self.model = DTQN(input_dim=input_dim, output_dim=1, k=self.k).to(self.device)
            self.target_model = DTQN(input_dim=input_dim, output_dim=1, k=self.k).to(self.device)
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=DTQN_LR, eps=1e-5)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)
            if os.path.isfile(self.checkpoint_path):
                try:
                    ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
                    if isinstance(ckpt, dict) and 'model' in ckpt:
                        self.model.load_state_dict(ckpt['model'])
                        self.target_model.load_state_dict(ckpt.get('target_model', ckpt['model']))
                        if 'optimizer' in ckpt:
                            self.optimizer.load_state_dict(ckpt['optimizer'])
                        if 'scheduler' in ckpt:
                            self.scheduler.load_state_dict(ckpt['scheduler'])
                        if 'normalizer' in ckpt:
                            self.normalizer.load_state_dict(ckpt['normalizer'])
                        if 'eps' in ckpt:
                            self.eps = ckpt['eps']
                        if 'update_count' in ckpt:
                            self.update_count = ckpt['update_count']
                    else:
                        self.model.load_state_dict(ckpt)
                        self.target_model.load_state_dict(ckpt)
                except Exception:
                    pass
            self.model.eval()

    def _soft_update_target(self):
        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def select_subgoal(self, state, candidates, mask):
        agent = state['agent']
        enemies = state['enemies']
        goal = state['goal']
        obstacles_shapely = state['obstacles_shapely']

        valid_indices = [i for i, m in enumerate(mask) if m]
        if not valid_indices:
            return -1

        # Encode all candidates — cache agent visibility result for reuse
        cand_indices = []
        feature_vecs = []
        seqs = []
        for i in valid_indices:
            v = self.encode_state_candidate(agent, enemies, candidates[i], goal, obstacles_shapely)
            self._ensure_model(len(v))
            seq = self.obs_history.get_sequence(v)
            cand_indices.append(i)
            feature_vecs.append(v)
            seqs.append(seq)

        # Batch forward pass — single GPU call for all candidates
        with torch.no_grad():
            batch = torch.from_numpy(np.stack(seqs, axis=0)).float().to(self.device)
            q_all = self.model(batch)[:, -1, 0]  # (num_candidates,)
            q_values = q_all.cpu().tolist()

        if np.random.rand() < self.eps:
            chosen_local = int(np.random.randint(len(cand_indices)))
        else:
            chosen_local = int(np.argmax(q_values))

        chosen_global_idx = cand_indices[chosen_local]
        chosen_q = q_values[chosen_local]

        chosen_vec = feature_vecs[chosen_local]
        chosen_seq = self.obs_history.get_sequence(chosen_vec)
        self.obs_history.push(chosen_vec)

        self.last_q_values = list(zip(cand_indices, q_values))
        self.last_selected = chosen_global_idx
        self.last_selected_q = chosen_q
        self.last_chosen_seq = chosen_seq.copy()

        self.decision_count += 1
        if self.train_mode:
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

        return chosen_global_idx

    def update(self, s_vec, R, gamma_k, next_candidates, next_mask, agent, enemies, goal, obstacles_shapely, done):
        self._ensure_model(len(s_vec) if self.input_dim is None else self.input_dim)

        # --- Compute best next-state sequence for replay buffer ---
        best_next_seq = None
        max_next_q = 0.0
        if not done:
            valid_next = [i for i, m in enumerate(next_mask) if m]
            if valid_next:
                next_seqs = []
                for i in valid_next:
                    v_next = self.encode_state_candidate(agent, enemies, next_candidates[i], goal, obstacles_shapely)
                    seq_next = self.obs_history.get_sequence(v_next)
                    next_seqs.append(seq_next.copy())

                # Batch inference: single forward pass for all next-state candidates
                with torch.no_grad():
                    next_batch = torch.from_numpy(np.stack(next_seqs, axis=0)).float().to(self.device)
                    q_online_all = self.model(next_batch)[:, -1, 0]
                    best_idx = int(q_online_all.argmax().item())
                    best_next_seq = next_seqs[best_idx]
                    # Evaluate best action with target network (double DQN)
                    x_best = next_batch[best_idx:best_idx+1]
                    max_next_q = self.target_model(x_best)[:, -1, 0].item()

        # --- Store full transition in replay buffer (NOT precomputed targets) ---
        if self.last_chosen_seq is not None:
            self.replay_buffer.push(
                self.last_chosen_seq, R, gamma_k, best_next_seq, done)

        # --- Online update with fresh TD target ---
        pred_value = 0.0
        td_target = R + gamma_k * max_next_q if not done else R
        target_value = td_target
        online_loss = 0.0

        if self.last_chosen_seq is not None:
            self.model.train()
            x = torch.from_numpy(self.last_chosen_seq).float().to(self.device).unsqueeze(0)
            q_pred = self.model(x)[:, -1, 0]
            target_t = torch.tensor([td_target], dtype=torch.float32, device=self.device)
            loss = F.smooth_l1_loss(q_pred, target_t)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.model.eval()
            pred_value = float(q_pred.detach().cpu().item())
            online_loss = float(loss.detach().cpu().item())

        batch_loss = 0.0
        if self.train_mode and self.decision_count % self.train_every == 0:
            batch_loss = self._train_from_buffer()

        self.update_count += 1
        self._soft_update_target()
        if self.scheduler:
            self.scheduler.step()

        if (self.update_count % DTQN_SAVE_INTERVAL) == 0:
            self.save_checkpoint()

        return online_loss + batch_loss, pred_value, target_value

    def _train_from_buffer(self):
        if len(self.replay_buffer) < self.min_buffer_size:
            return 0.0

        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        if not samples:
            return 0.0

        # Unpack full transitions: (state_seq, R, gamma_k, next_best_seq, done)
        state_seqs = torch.stack([torch.from_numpy(s[0]).float() for s in samples]).to(self.device)
        rewards = torch.tensor([s[1] for s in samples], dtype=torch.float32, device=self.device)
        gamma_ks = torch.tensor([s[2] for s in samples], dtype=torch.float32, device=self.device)
        weights_t = torch.from_numpy(weights).float().to(self.device)

        # Recompute TD targets using CURRENT target network (not stale stored values)
        with torch.no_grad():
            targets = rewards.clone()
            non_done_indices = []
            non_done_next_seqs = []
            for i, s in enumerate(samples):
                if not s[4] and s[3] is not None:  # not done and has next_seq
                    non_done_indices.append(i)
                    non_done_next_seqs.append(torch.from_numpy(s[3]).float())

            if non_done_next_seqs:
                next_batch = torch.stack(non_done_next_seqs).to(self.device)
                q_nexts = self.target_model(next_batch)[:, -1, 0]
                for j, idx in enumerate(non_done_indices):
                    targets[idx] = rewards[idx] + gamma_ks[idx] * q_nexts[j]

        self.model.train()
        q_preds = self.model(state_seqs)[:, -1, 0]
        td_errors = (q_preds - targets).detach().cpu().numpy()
        losses = self.criterion(q_preds, targets)
        loss = (weights_t * losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.model.eval()

        self.replay_buffer.update_priorities(indices, td_errors)
        return float(loss.detach().cpu().item())

    def save_checkpoint(self):
        if self.model is None:
            return
        try:
            ckpt = {
                'model': self.model.state_dict(),
                'target_model': self.target_model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'normalizer': self.normalizer.state_dict(),
                'eps': self.eps,
                'update_count': self.update_count,
            }
            torch.save(ckpt, self.checkpoint_path)
        except Exception:
            pass

    def reset_episode(self):
        """Reset per-episode state (call at start of each new episode)."""
        self.obs_history.reset()
        self.last_chosen_seq = None

def _init_logging():
    try:
        with open(LOG_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'decision_idx', 'agent_x', 'agent_y', 'chosen_idx', 'subgoal_x', 'subgoal_y',
                'R', 'k', 'termination', 'q_selected', 'num_valid', 'timestamp'
            ])
    except Exception:
        pass

def _log_decision(decision_idx, agent_pos, chosen_idx, subgoal_centroid, R, k, termination, q_selected, num_valid):
    try:
        with open(LOG_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                decision_idx, agent_pos[0], agent_pos[1], chosen_idx, subgoal_centroid[0], subgoal_centroid[1],
                R, k, termination, (q_selected if q_selected is not None else ''), num_valid, int(math.floor(threading.get_native_id()))
            ])
    except Exception:
        pass

def _init_train_metrics():
    try:
        out_dir = os.path.join(_BASE_DIR, TRAIN_METRICS_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        need_header = (not os.path.isfile(TRAIN_METRICS_PATH)) or (os.path.getsize(TRAIN_METRICS_PATH) == 0)
        if need_header:
            with open(TRAIN_METRICS_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'episode', 'decision_idx', 'update_count', 'loss', 'q_pred', 'target', 'R', 'k', 'termination', 'timestamp'
                ])
    except Exception:
        pass

def _log_train_metric(episode_idx, decision_idx, update_count, loss_value, q_pred_value, target_value, R, k, termination):
    try:
        with open(TRAIN_METRICS_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                (episode_idx if episode_idx is not None else ''),
                decision_idx,
                update_count,
                loss_value,
                q_pred_value,
                target_value,
                R,
                k,
                termination,
                int(math.floor(threading.get_native_id()))
            ])
    except Exception:
        pass

def build_candidate_obstacles(obstacles):
    """
    Convert obstacles list of tuples (poly, bounds, centroid) to a list of
    candidate dicts with stable index.
    """
    candidates = []
    for idx, (poly, _bounds, centroid) in enumerate(obstacles):
        candidates.append({'index': idx, 'poly': poly, 'centroid': centroid})
    return candidates

def build_high_level_state(agent, enemies, candidates, goal, obs_shapely):
    """
    Build a compact feature dict for high-level policy. For heuristic we only need goal.
    """
    return {
        'agent_pos': agent.pos.copy(),
        'agent_goal': agent.goal.copy(),
        'goal': goal.copy(),
        'num_enemies': len(enemies),
        'num_candidates': len(candidates),
        'agent': agent,
        'enemies': enemies,
        'obstacles_shapely': obs_shapely,
    }

def compute_action_mask(candidates, agent, goal, min_dist_same=2.0):
    """
    Mask out obstacles that are too close to current agent position, or that
    are farther from the agent than the final goal (to avoid going against goal).
    Returns np.ndarray[bool] of length len(candidates).
    """
    mask = []
    agent_to_goal = math.sqrt((agent.pos[0] - goal[0])**2 + (agent.pos[1] - goal[1])**2)
    for c in candidates:
        centroid = c['centroid']
        dist_agent_to_centroid = math.sqrt((agent.pos[0] - centroid[0])**2 + (agent.pos[1] - centroid[1])**2)
        # Do not go to the same obstacle (too close)
        if dist_agent_to_centroid < min_dist_same:
            mask.append(False)
            continue
        # Do not pick if it is farther than the goal (avoid going away from goal)
        if agent_to_goal < dist_agent_to_centroid:
            mask.append(False)
            continue
        mask.append(True)
    return np.array(mask, dtype=bool)

def any_enemy_sees(agent_pos, enemies, obstacles_shapely):
    for enemy in enemies:
        if enemy.can_see(agent_pos, obstacles_shapely):
            return True
    return False

def agent_inside_any_obstacle(agent_pos, obstacles_shapely):
    p = Point(agent_pos)
    for obs in obstacles_shapely:
        if obs.contains(p):
            return True
    return False

def run_option_navigate_to(agent, enemies, obstacles, obstacles_shapely, subgoal_centroid, cfg, arrays_ctx):
    """
    Execute low-level navigation for up to MAX_K_PER_OPTION steps toward subgoal_centroid.
    Returns: (R, k, termination_reason)
    arrays_ctx: dict with references to agent_positions, enemy_positions, enemy_directions for visualization logging
    cfg must contain: MAX_K_PER_OPTION, dt, agent_goal, past_agent_positions
    """
    R = 0.0
    k = 0
    termination_reason = None

    _dt = cfg['dt']
    _agent_goal = cfg['agent_goal']
    _past_positions = cfg['past_agent_positions']

    while k < cfg['MAX_K_PER_OPTION']:
        for i, enemy in enumerate(enemies):
            enemy.update(_dt, obstacles_shapely)
            arrays_ctx['enemy_positions'][i].append(enemy.pos.copy())
            arrays_ctx['enemy_directions'][i].append(enemy.direction.copy())

        prev_dist = math.sqrt((agent.pos[0] - subgoal_centroid[0])**2 + (agent.pos[1] - subgoal_centroid[1])**2)
        prev_goal_dist = math.sqrt((agent.pos[0] - _agent_goal[0])**2 + (agent.pos[1] - _agent_goal[1])**2)
        agent.goal = np.array(subgoal_centroid, dtype=float)

        to_subgoal = np.array(subgoal_centroid, dtype=float) - agent.pos
        to_subgoal_norm = np.linalg.norm(to_subgoal)
        step_dir = (to_subgoal / to_subgoal_norm) if to_subgoal_norm > 1e-6 else np.zeros(2)
        probe_point = agent.pos + 0.5 * step_dir
        probe_in_view = any_enemy_sees(probe_point, enemies, obstacles_shapely)
        min_enemy_dist = min(np.linalg.norm(agent.pos - enemy.pos) for enemy in enemies) if enemies else float('inf')
        proximity_risk = (min_enemy_dist < 2.0)
        should_wait = probe_in_view or proximity_risk

        F_desired = (np.zeros(2) if should_wait else (desired_force(agent, desired_speed=1.0) * 10.0))
        F_obs = obstacle_force(agent, obstacles_shapely, threshold=1.0, repulsive_coeff=100.0)

        F_enemy = np.zeros(2)
        F_anticipatory = np.zeros(2)
        agent_in_any_fov = False
        for enemy in enemies:
            if enemy.can_see(agent.pos, obstacles_shapely):
                agent_in_any_fov = True
                diff = agent.pos - enemy.pos
                d = np.linalg.norm(diff)
                if d > 0:
                    F_enemy += normalize(diff) * (300.0 / ((d + 0.1)**2))
            F_anticipatory += anticipatory_enemy_avoidance_force(agent, enemy, T=10.0, weight=1500.0)

        _past_positions.append(agent.pos.copy())
        if len(_past_positions) > 100:
            _past_positions.pop(0)
        stuck = is_agent_stuck(agent, _past_positions, threshold=0.1, window=100)
        if stuck:
            escape_dir = find_escape_direction(agent, obstacles_shapely, _past_positions)
            F_escape_stuck = escape_dir * 25.0
        else:
            F_escape_stuck = np.zeros(2)

        F_total = F_desired + F_obs + F_enemy + F_anticipatory + F_escape_stuck

        in_view = agent_in_any_fov
        agent.update(F_total, _dt, smoothing=0.9, max_speed=(1.0 if not in_view else 2.0))
        arrays_ctx['agent_positions'].append(agent.pos.copy())

        curr_dist = math.sqrt((agent.pos[0] - subgoal_centroid[0])**2 + (agent.pos[1] - subgoal_centroid[1])**2)
        curr_goal_dist = math.sqrt((agent.pos[0] - _agent_goal[0])**2 + (agent.pos[1] - _agent_goal[1])**2)
        r = 0.0
        r += REWARD_WEIGHTS['progress_to_subgoal'] * (prev_dist - curr_dist)
        r += REWARD_WEIGHTS['goal_progress'] * (prev_goal_dist - curr_goal_dist)

        agent_exposed = agent_in_any_fov  # Reuse from force computation above
        if agent_exposed:
            exposure_scale = max(0.5, 1.0 - min_enemy_dist / 5.0) if enemies else 1.0
            r -= REWARD_WEIGHTS['exposure_penalty'] * exposure_scale
        if agent_inside_any_obstacle(agent.pos, obstacles_shapely):
            r -= REWARD_WEIGHTS['collision_penalty']
        if should_wait and not agent_exposed:
            r -= REWARD_WEIGHTS['time_penalty'] * 0.25
        else:
            r -= REWARD_WEIGHTS['time_penalty']

        r = np.clip(r, -10.0, 10.0)

        R += (GAMMA ** k) * r
        k += 1

        if curr_dist < 2.0 and termination_reason is None:
            R += (GAMMA ** k) * REWARD_WEIGHTS['subgoal_bonus']
            termination_reason = 'SUBGOAL_REACHED'
        goal_dist = math.sqrt((agent.pos[0] - _agent_goal[0])**2 + (agent.pos[1] - _agent_goal[1])**2)
        if goal_dist <= 2.0:
            R += (GAMMA ** k) * REWARD_WEIGHTS['goal_bonus']
            termination_reason = 'GOAL_REACHED'

        if termination_reason in ('SUBGOAL_REACHED', 'GOAL_REACHED'):
            break

    return R, k, (termination_reason or 'MAX_STEPS')


def save_episode_video(agent_positions, enemy_positions, enemy_directions, enemies, obstacles_shapely, obstacles, out_path):
    """Save an episode trajectory as an MP4 video."""
    agent_positions = np.array(agent_positions)
    num_enemies = len(enemies)
    if enemy_positions and len(enemy_positions[0]) < len(agent_positions):
        for i in range(num_enemies):
            if len(enemy_positions[i]) > 0:
                while len(enemy_positions[i]) < len(agent_positions):
                    enemy_positions[i].append(enemy_positions[i][-1])
                    enemy_directions[i].append(enemy_directions[i][-1])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(scene_min - 2, scene_max + 2)
    ax.set_ylim(scene_min - 2, scene_max + 2)
    ax.set_aspect('equal')
    ax.set_title("Stealth Simulation - Training Episode")

    for obs_data in obstacles:
        poly, _, _ = obs_data
        coords = np.array(poly.exterior.coords)
        ax.add_patch(MplPolygon(coords, closed=True, color='gray', alpha=0.5))

    enemy_dots = []
    enemy_vis_patches = []
    for enemy in enemies:
        dot, = ax.plot(enemy.pos[0], enemy.pos[1], 'ro')
        enemy_dots.append(dot)
        vis_poly = enemy.get_visibility_polygon(obstacles_shapely)
        patch = MplPolygon(np.array(vis_poly.exterior.coords), closed=True, color='red', alpha=0.2)
        ax.add_patch(patch)
        enemy_vis_patches.append(patch)

    line, = ax.plot([], [], 'b-', lw=2)
    agent_dot, = ax.plot([], [], 'bo', ms=8)

    def init():
        line.set_data([], [])
        agent_dot.set_data([], [])
        return line, agent_dot, *enemy_dots, *enemy_vis_patches

    def animate(i):
        line.set_data(agent_positions[:i, 0], agent_positions[:i, 1])
        agent_dot.set_data([agent_positions[i, 0]], [agent_positions[i, 1]])
        for j in range(num_enemies):
            pos = enemy_positions[j][i]
            enemy_dots[j].set_data([pos[0]], [pos[1]])
            dvec = enemy_directions[j][i]
            vis_poly = compute_visibility_polygon_raycast(pos, dvec, enemies[j].fov, enemies[j].view_range, obstacles_shapely, num_rays=100)
            enemy_vis_patches[j].set_xy(np.array(vis_poly.exterior.coords))
        return line, agent_dot, *enemy_dots, *enemy_vis_patches

    ani = animation.FuncAnimation(fig, animate, frames=len(agent_positions), interval=50, blit=True, init_func=init)
    if VIDEO_WRITER:
        ani.save(out_path, writer=VIDEO_WRITER, fps=20)
    else:
        # Fallback: let matplotlib auto-detect
        ani.save(out_path, fps=20)
    plt.close(fig)


# --------------------------
# Module-level constants
# --------------------------
dt = 0.25
sim_time = 60.0
steps = int(sim_time / dt)


def run_training_episode(policy, episode_idx=None, verbose=False, save_video=False, video_out_dir=None):
    """Run a single training episode with the given high-level policy.
    The policy (and its replay buffer, normalizer, etc.) persists across calls.
    Returns a dict of episode results.
    If save_video is True, saves an MP4 to video_out_dir (default: train/videos).
    """
    _agent = Agent(np.array([scene_min, scene_min]), np.array([scene_max, scene_max]), max_speed=1.0)
    _agent_goal = np.array([scene_max, scene_max])

    _enemies = []
    for _ in range(5):
        pos = np.random.uniform(scene_min + 2, scene_max, size=2)
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        fov = np.pi / 3 + np.random.uniform(-0.1, 0.1)
        _enemies.append(EnemyAgent(pos, direction, fov, speed=0.3))

    _obstacles = []
    _obstacles_shapely = []
    for _ in range(6):
        poly = create_random_convex_polygon(sides=5)
        _obstacles_shapely.append(poly)
        bounds = poly.bounds
        centroid = get_polygon_centroid(poly)
        _obstacles.append((poly, bounds, centroid))
    goal_poly = Polygon(
        [(_agent_goal[0] - 0.5, _agent_goal[1] - 0.5),
         (_agent_goal[0] + 0.5, _agent_goal[1] - 0.5),
         (_agent_goal[0] + 0.5, _agent_goal[1] + 0.5),
         (_agent_goal[0] - 0.5, _agent_goal[1] + 0.5)])
    _obstacles_shapely.append(goal_poly)
    _obstacles.append((goal_poly, goal_poly.bounds, get_polygon_centroid(goal_poly)))

    _past_positions = []

    agent_positions = [_agent.pos.copy()]
    enemy_positions = [[e.pos.copy()] for e in _enemies]
    enemy_directions = [[e.direction.copy()] for e in _enemies]

    if hasattr(policy, 'reset_episode'):
        policy.reset_episode()

    total_R = 0.0
    num_decisions = 0
    max_decisions = 100

    goal_reached = False
    while (np.linalg.norm(_agent.pos - _agent_goal) > 2.0) and num_decisions < max_decisions:
        candidates = build_candidate_obstacles(_obstacles)
        mask = compute_action_mask(candidates, _agent, _agent_goal)
        state = build_high_level_state(_agent, _enemies, candidates, _agent_goal, _obstacles_shapely)

        chosen_idx = policy.select_subgoal(state, candidates, mask)
        if chosen_idx == -1:
            subgoal_centroid = _agent_goal.copy()
            chosen_state_vec = None
        else:
            subgoal_centroid = candidates[chosen_idx]['centroid']
            if isinstance(policy, DTQNPolicy):
                chosen_state_vec = policy.encode_state_candidate(
                    _agent, _enemies, candidates[chosen_idx], _agent_goal, _obstacles_shapely)
            else:
                chosen_state_vec = None

        arrays_ctx = {'agent_positions': agent_positions, 'enemy_positions': enemy_positions, 'enemy_directions': enemy_directions}
        cfg = {
            'MAX_K_PER_OPTION': MAX_K_PER_OPTION,
            'dt': dt,
            'agent_goal': _agent_goal,
            'past_agent_positions': _past_positions,
        }

        R, k, termination = run_option_navigate_to(
            _agent, _enemies, _obstacles, _obstacles_shapely, subgoal_centroid, cfg, arrays_ctx)
        total_R += R
        num_decisions += 1

        done = (termination == 'GOAL_REACHED') or (num_decisions >= max_decisions)

        # Terminal failure penalty: give the Q-network a clear signal that
        # running out of decisions without reaching the goal is very bad.
        if done and termination != 'GOAL_REACHED':
            R -= REWARD_WEIGHTS['failure_penalty']
            total_R -= REWARD_WEIGHTS['failure_penalty']

        if TRAIN_HIGH_LEVEL and isinstance(policy, DTQNPolicy) and chosen_idx != -1 and chosen_state_vec is not None:
            next_candidates = build_candidate_obstacles(_obstacles)
            next_mask = compute_action_mask(next_candidates, _agent, _agent_goal)
            gamma_k = GAMMA ** k
            loss_val, q_pred, q_target = policy.update(
                s_vec=chosen_state_vec, R=R, gamma_k=gamma_k,
                next_candidates=next_candidates, next_mask=next_mask,
                agent=_agent, enemies=_enemies, goal=_agent_goal,
                obstacles_shapely=_obstacles_shapely, done=done)
            _log_train_metric(
                episode_idx, num_decisions, policy.update_count,
                loss_val, q_pred, q_target, R, k, termination)

        if termination == 'GOAL_REACHED':
            goal_reached = True
            break

    if np.linalg.norm(_agent.pos - _agent_goal) <= 2.0:
        goal_reached = True

    if save_video and len(agent_positions) > 1:
        out_dir = video_out_dir or os.path.join(_BASE_DIR, 'train', 'videos')
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ep_suffix = f"_ep{episode_idx}" if episode_idx is not None else ""
        filename = f"animation_{ts}{ep_suffix}.mp4"
        out_path = os.path.join(out_dir, filename)
        try:
            save_episode_video(agent_positions, enemy_positions, enemy_directions, _enemies, _obstacles_shapely, _obstacles, out_path)
        except Exception as e:
            pass

    return {
        'success': goal_reached,
        'decisions': num_decisions,
        'total_reward': total_R,
        'termination': 'GOAL_REACHED' if goal_reached else 'MAX_DECISIONS',
        'final_pos': _agent.pos.copy(),
        'final_goal_dist': float(np.linalg.norm(_agent.pos - _agent_goal)),
    }


# --------------------------
# Direct-run Simulation Setup (backward compat)
# --------------------------
if __name__ == '__main__':
    agent_start = np.array([scene_min, scene_min])
    agent_goal = np.array([scene_max, scene_max])
    agent = Agent(agent_start, agent_goal, max_speed=1.0)

    num_enemies = 5
    enemies = []
    enemies_data = []
    for _ in range(num_enemies):
        pos = np.random.uniform(scene_min + 2, scene_max, size=2)
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        fov = np.pi / 3 + np.random.uniform(-0.1, 0.1)
        enemies_data.append(EnemyAgent(pos, direction, fov, speed=0.3))
        enemies.append(EnemyAgent(pos, direction, fov, speed=0.3))

    num_obstacles = 6
    obstacles = []
    obstacles_shapely = []
    past_agent_positions = []

    for _ in range(num_obstacles):
        poly = create_random_convex_polygon(sides=5)
        obstacles_shapely.append(poly)
        bounds = poly.bounds
        centroid = get_polygon_centroid(poly)
        obstacles.append((poly, bounds, centroid))

    goal_poly = Polygon([(agent_goal[0] - 0.5, agent_goal[1] - 0.5),
                         (agent_goal[0] + 0.5, agent_goal[1] - 0.5),
                         (agent_goal[0] + 0.5, agent_goal[1] + 0.5),
                         (agent_goal[0] - 0.5, agent_goal[1] + 0.5)])
    obstacles_shapely.append(goal_poly)
    bounds = goal_poly.bounds
    centroid = get_polygon_centroid(goal_poly)
    obstacles.append((goal_poly, bounds, centroid))

    agent_positions = [agent.pos.copy()]
    enemy_positions = [[] for _ in range(num_enemies)]
    enemy_directions = [[] for _ in range(num_enemies)]

    for i in range(num_enemies):
        enemy_positions[i].append(enemies[i].pos.copy())
        enemy_directions[i].append(enemies[i].direction.copy())

    stop_loop = False

    def wait_for_key():
        global stop_loop
        input("Press Enter to stop the loop...\n")
        stop_loop = True

    _key_listener = threading.Thread(target=wait_for_key, daemon=True)
    _key_listener.start()

    policy = DTQNPolicy() if USE_DTQN else HeuristicPolicy()
    if isinstance(policy, DTQNPolicy):
        _init_logging()
        _init_train_metrics()
        atexit.register(policy.save_checkpoint)

    while math.sqrt(abs(agent.pos[0] - agent_goal[0]) ** 2 + abs(agent.pos[1] - agent_goal[1]) ** 2) > 2 and not stop_loop:
        candidates = build_candidate_obstacles(obstacles)
        mask = compute_action_mask(candidates, agent, agent_goal)
        state = build_high_level_state(agent, enemies, candidates, agent_goal, obstacles_shapely)

        chosen_idx = policy.select_subgoal(state, candidates, mask)
        if chosen_idx == -1:
            subgoal_centroid = agent_goal.copy()
            chosen_state_vec = None
        else:
            subgoal_centroid = candidates[chosen_idx]['centroid']
            if isinstance(policy, DTQNPolicy):
                chosen_state_vec = policy.encode_state_candidate(agent, enemies, candidates[chosen_idx], agent_goal, obstacles_shapely)

        arrays_ctx = {
            'agent_positions': agent_positions,
            'enemy_positions': enemy_positions,
            'enemy_directions': enemy_directions,
        }
        cfg = {
            'MAX_K_PER_OPTION': MAX_K_PER_OPTION,
            'dt': dt,
            'agent_goal': agent_goal,
            'past_agent_positions': past_agent_positions,
        }

        R, k, termination = run_option_navigate_to(
            agent, enemies, obstacles, obstacles_shapely, subgoal_centroid, cfg, arrays_ctx)

        if isinstance(policy, DTQNPolicy):
            q_sel = getattr(policy, 'last_selected_q', None)
            num_valid = int(mask.sum())
            _log_decision(decision_idx=len(agent_positions) - 1, agent_pos=agent.pos.copy(), chosen_idx=chosen_idx,
                          subgoal_centroid=subgoal_centroid, R=R, k=k, termination=termination,
                          q_selected=q_sel, num_valid=num_valid)

        done = (termination == 'GOAL_REACHED') or stop_loop
        if TRAIN_HIGH_LEVEL and isinstance(policy, DTQNPolicy) and chosen_idx != -1 and chosen_state_vec is not None:
            next_candidates = build_candidate_obstacles(obstacles)
            next_mask = compute_action_mask(next_candidates, agent, agent_goal)
            gamma_k = (GAMMA ** k)
            loss_value, q_pred_value, target_value = policy.update(
                s_vec=chosen_state_vec, R=R, gamma_k=gamma_k,
                next_candidates=next_candidates, next_mask=next_mask,
                agent=agent, enemies=enemies, goal=agent_goal,
                obstacles_shapely=obstacles_shapely, done=done)
            try:
                ep_env = os.environ.get('EPISODE_INDEX')
                ep_idx = int(ep_env) if (ep_env is not None and ep_env.isdigit()) else None
            except Exception:
                ep_idx = None
            _log_train_metric(
                episode_idx=ep_idx, decision_idx=(len(agent_positions) - 1),
                update_count=policy.update_count, loss_value=loss_value,
                q_pred_value=q_pred_value, target_value=target_value,
                R=R, k=k, termination=termination)

        if termination == 'GOAL_REACHED':
            print('Close enough to final goal')
            break
        if termination == 'SUBGOAL_REACHED':
            print('Reached subgoal')
            continue

    agent_positions = np.array(agent_positions)
    if enemy_positions and len(enemy_positions[0]) < len(agent_positions):
        for i in range(num_enemies):
            if len(enemy_positions[i]) > 0:
                while len(enemy_positions[i]) < len(agent_positions):
                    enemy_positions[i].append(enemy_positions[i][-1])
                    enemy_directions[i].append(enemy_directions[i][-1])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(scene_min - 2, scene_max + 2)
    ax.set_ylim(scene_min - 2, scene_max + 2)
    ax.set_aspect('equal')
    ax.set_title("Stealth Simulation with Enhanced Anticipation")

    for obs_data in obstacles:
        poly, _, _ = obs_data
        coords = np.array(poly.exterior.coords)
        ax.add_patch(MplPolygon(coords, closed=True, color='gray', alpha=0.5))

    enemy_dots = []
    enemy_vis_patches = []
    for enemy in enemies:
        dot, = ax.plot(enemy.pos[0], enemy.pos[1], 'ro')
        enemy_dots.append(dot)
        vis_poly = enemy.get_visibility_polygon(obstacles_shapely)
        patch = MplPolygon(np.array(vis_poly.exterior.coords), closed=True, color='red', alpha=0.2)
        ax.add_patch(patch)
        enemy_vis_patches.append(patch)

    line, = ax.plot([], [], 'b-', lw=2)
    agent_dot, = ax.plot([], [], 'bo', ms=8)

    def init():
        line.set_data([], [])
        agent_dot.set_data([], [])
        return line, agent_dot, *enemy_dots, *enemy_vis_patches

    def animate(i):
        line.set_data(agent_positions[:i, 0], agent_positions[:i, 1])
        agent_dot.set_data([agent_positions[i, 0]], [agent_positions[i, 1]])
        for j in range(num_enemies):
            pos = enemy_positions[j][i]
            enemy_dots[j].set_data([pos[0]], [pos[1]])
            dvec = enemy_directions[j][i]
            vis_poly = compute_visibility_polygon_raycast(pos, dvec, enemies[j].fov, enemies[j].view_range, obstacles_shapely, num_rays=100)
            enemy_vis_patches[j].set_xy(np.array(vis_poly.exterior.coords))
        return line, agent_dot, *enemy_dots, *enemy_vis_patches

    ani = animation.FuncAnimation(fig, animate, frames=len(agent_positions),
                                  interval=50, blit=True, init_func=init)
    plt.close()

    if not DISABLE_VIDEO:
        video_subdir = os.environ.get('VIDEO_SUBDIR', 'eval')
        out_dir = os.path.join(_BASE_DIR, video_subdir)
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ep_idx = os.environ.get('EPISODE_INDEX')
        ep_suffix = f"_ep{ep_idx}" if ep_idx is not None else ""
        filename = f"animation_{ts}{ep_suffix}.mp4"
        out_path = os.path.join(out_dir, filename)
        if VIDEO_WRITER:
            ani.save(out_path, writer=VIDEO_WRITER, fps=20)
        else:
            # Fallback: let matplotlib auto-detect
            ani.save(out_path, fps=20)
        Video(out_path, embed=True)