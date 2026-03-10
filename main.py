import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib import animation
import copy
import threading
import math
import random
from ros_data_local import ROSDataLocal
import subprocess
import time
from collections import defaultdict



# --------------------------
# Simulation Setup
# --------------------------
dt = 0.25                     # Time step (seconds)
sim_time = 60.0               # Total simulation time (seconds)
steps = int(sim_time / dt)    # Total number of simulation steps

# Main agent starts at robot's current position (0,0 in robot frame) and has a goal at the far corner of the environment
agent_start = np.array([0.0, 0.0])  # Robot's current position in robot frame
NUM_ENEMIES = 2  # Number of enemy huskies (husky1, husky2)
FIXED_ENEMY_SPAWNS_ROBOT = [  # Easy to change fixed robot-frame spawns
    np.array([5.0, 7.0]),
    np.array([10.0, 11.0]),
]

# --------------------------
# Visualization Functions
# --------------------------
def save_pointcloud_visualization(points, agent_pos, filename="pc.png"):
    """
    Save point cloud visualization to PNG file with points centered around the robot (red dot at center)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Always render in robot-centric view with the robot at the origin.
    robot_center = np.array([0.0, 0.0])
    view_range = 10  # Show 10 units around robot in each direction
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Point Cloud Data (Robot-Centered View)\nRobot at center (0.0, 0.0), World pos: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})")
    ax.grid(True, alpha=0.3)
    
    # Disable autoscaling so artists do not change our fixed limits later
    ax.set_autoscale_on(False)
    
    # Plot points relative to current robot position if provided
    if points:
        points_array = np.array(points)
        if points_array.shape[1] >= 2:
            relative_points = points_array[:, :2] - agent_pos
            ax.scatter(relative_points[:, 0], relative_points[:, 1],
                      c='blue', s=10, alpha=0.7, label=f'Point Cloud ({len(points)} points)')
    
    # Plot robot at the origin
    ax.scatter(robot_center[0], robot_center[1], c='red', s=100, marker='o', label='Robot (center)')
    
    # Plot goal relative to robot
    goal_relative = agent_goal - agent_pos
    ax.scatter(goal_relative[0], goal_relative[1], c='green', s=100, marker='*',
               label=f'Goal (relative: {goal_relative[0]:.2f},{goal_relative[1]:.2f})')
    
    # Add crosshairs at the origin
    ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.3)
    ax.axvline(x=0.0, color='red', linestyle='--', alpha=0.3)
    
    # Finally, enforce fixed limits AFTER all plotting so nothing overrides them
    ax.set_xlim(-view_range, view_range)
    ax.set_ylim(-view_range, view_range)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Point cloud visualization saved as {filename}")

def save_obstacles_visualization(obstacles_data, agent_pos, filename="obs.png"):
    """
    Save obstacles visualization to PNG file with enemies and their FOV
    Handles Shapely Polygon objects
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(ENVIRONMENT_MIN-2, ENVIRONMENT_MAX+2)
    ax.set_ylim(ENVIRONMENT_MIN-2, ENVIRONMENT_MAX+2)
    ax.set_aspect('equal')
    ax.set_title(f"Obstacles from Point Cloud with Enemies\nEnvironment: ({ENVIRONMENT_MIN}, {ENVIRONMENT_MIN}) to ({ENVIRONMENT_MAX}, {ENVIRONMENT_MAX})")
    ax.grid(True, alpha=0.3)
    
    # Draw obstacles
    for i, obs in enumerate(obstacles_data):
        if hasattr(obs, 'exterior'):  # Shapely Polygon object
            coords = np.array(obs.exterior.coords)
            ax.add_patch(MplPolygon(coords, closed=True, color='gray', alpha=0.5, 
                                   label=f'Obstacle {i+1}' if i == 0 else ""))
            # Add obstacle number
            centroid = obs.centroid
            ax.text(centroid.x, centroid.y, f'{i+1}', fontsize=12, ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Draw enemies and their FOV (similar to older_main.py animation style)
    current_enemies = []
    if USE_ROS:
        # Get current enemy positions from ROS data
        for idx_enemy in range(NUM_ENEMIES):
            if idx_enemy in enemy_positions and idx_enemy in enemy_headings:
                pos = np.array(enemy_positions[idx_enemy])
                heading = enemy_headings[idx_enemy]
                direction = np.array([math.cos(heading), math.sin(heading)])
                current_enemies.append((pos, direction))
    else:
        # Get simulation enemies if available
        if hasattr(ros_data, 'sim_enemy_agents'):
            for enemy_agent in ros_data.sim_enemy_agents:
                current_enemies.append((enemy_agent.pos, enemy_agent.direction))
    
    # Draw each enemy and its FOV
    for idx, (pos, direction) in enumerate(current_enemies):
        # Filter out placeholder (0,0) positions if heading is also zero-like
        if np.allclose(pos, np.zeros(2), atol=1e-6):
            continue
        # Draw enemy as red circle (like older_main.py)
        ax.scatter(pos[0], pos[1], c='red', s=100, marker='o', 
                  label=f'Enemy {idx+1}' if idx == 0 else "", edgecolors='darkred', linewidth=2)
        
        # Create EnemyAgent to get FOV polygon
        enemy = EnemyAgent(pos, direction)
        vis_poly = enemy.get_visibility_polygon(obstacles_data)
        
        if hasattr(vis_poly, 'exterior'):
            coords = np.array(vis_poly.exterior.coords)
            # Draw FOV as red transparent polygon (like older_main.py)
            ax.add_patch(MplPolygon(coords, closed=True, color='red', alpha=0.2,
                                   label='Enemy FOV' if idx == 0 else ""))
        
        # Add enemy number label
        ax.text(pos[0], pos[1] - 0.5, f'E{idx+1}', fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.8, edgecolor="darkred"),
                color='white', weight='bold')
    
    # Plot robot position at agent_pos
    ax.scatter(agent_pos[0], agent_pos[1], c='blue', s=100, marker='o', label=f'Robot ({agent_pos[0]:.2f},{agent_pos[1]:.2f})',
              edgecolors='darkblue', linewidth=2)
    
    # Plot goal
    ax.scatter(agent_goal[0], agent_goal[1], c='green', s=100, marker='*', label='Goal',
              edgecolors='darkgreen', linewidth=2)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Obstacles visualization with enemies saved as {filename}")

# Shapely is used for geometric computations (polygons, intersections, etc.)
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union

# --------------------------
# Global Environment Configuration
# --------------------------
# Environment bounds in robot-centric coordinates
ENVIRONMENT_MIN = -5
ENVIRONMENT_MAX = 15

# --------------------------
# Scene Setup & Utility Functions
# --------------------------

# Define the simulation boundaries using global variables
scene_min, scene_max = ENVIRONMENT_MIN, ENVIRONMENT_MAX

# Main agent goal at the far corner of the environment
agent_goal = np.array([ENVIRONMENT_MAX - 1, ENVIRONMENT_MAX - 1])  # Goal near the far corner of environment

# Toggle between ROS and simulation mode
USE_ROS = True  # Set to True to use ROS, False for 2D simulation

# Toggle enemies on/off
ENABLE_ENEMIES = True  # Set to True to enable enemies, False to disable them

# Toggle between simple and anticipatory policy
USE_ANTICIPATORY_POLICY = True  # Set to True to use the advanced subgoal/anticipatory logic

# Initialize ROS data handler
ros_data = ROSDataLocal(scene_min=ENVIRONMENT_MIN, scene_max=ENVIRONMENT_MAX, use_ros=USE_ROS, num_enemies=NUM_ENEMIES)

def normalize(v):
    """
    Returns the unit vector in the direction of vector v.
    """
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def velocity_to_force(linear_vel, angular_vel, current_direction):
    """
    Convert linear and angular velocity to a force vector in 2D space.
    """
    # Calculate the new direction based on angular velocity
    angle = angular_vel * dt  # dt is the time step
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    new_direction = normalize(np.dot(rot, current_direction))
    
    # Calculate the force vector
    force = linear_vel * new_direction
    return force, new_direction

def force_to_velocity(force, current_direction):
    """
    Convert a force vector to linear and angular velocity commands.
    """
    # Calculate linear velocity (magnitude of force)
    linear_vel = np.linalg.norm(force)
    
    # Calculate angular velocity based on the angle between current direction and force
    if linear_vel > 0:
        force_direction = normalize(force)
        angle = math.acos(np.clip(np.dot(current_direction, force_direction), -1.0, 1.0))
        # Determine the sign of the angle using cross product
        cross_product = np.cross(current_direction, force_direction)
        angular_vel = angle * (1.0 if cross_product > 0 else -1.0) / dt
    else:
        angular_vel = 0.0
    
    return linear_vel, angular_vel

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

def find_escape_direction(agent, obstacles_data, past_positions, num_samples=16):
    """
    Finds a direction to escape when the agent is stuck.
    
    Parameters:
        agent: The agent that is stuck
        obstacles_data: List of obstacle polygons
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
    angles = np.linspace(0, 2*math.pi, num_samples, endpoint=False)
    directions = []
    clearances = []
    
    for angle in angles:
        # Generate a direction vector for this angle
        direction = np.array([math.cos(angle), math.sin(angle)])
        
        # Skip directions that would lead back to where we came from
        dot_product = np.dot(direction, avoid_dir)
        if dot_product < -0.7:  # Avoid going directly back
            continue
            
        # Check how far we can go in this direction before hitting an obstacle
        clearance = float('inf')
        ray = LineString([agent.pos, agent.pos + direction * 10])  # Create a long ray
        
        for obs in obstacles_data:
            if hasattr(obs, 'intersection'):  # Shapely Polygon object
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
            direction = np.array([math.cos(angle), math.sin(angle)])
            directions.append(direction)
            
            clearance = float('inf')
            ray = LineString([agent.pos, agent.pos + direction * 10])
            
            for obs in obstacles_data:
                if hasattr(obs, 'intersection'):  # Shapely Polygon object
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
    Handles Shapely Polygon objects.
    """
    force = np.zeros(2)
    entity_point = Point(entity.pos)
    
    for obs in obstacles:
        if hasattr(obs, 'distance'):  # Shapely Polygon object
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
    Computes the field-of-view polygon for an enemy using ray-casting.
    Includes both primary FOV (forward sector) and secondary FOV (complete circle with reduced range).

    Parameters:
        pos         : Enemy's position as a numpy array.
        direction   : Enemy's forward direction (unit vector).
        fov         : Field of view in radians for the primary sector.
        view_range  : Maximum distance the enemy can see in primary FOV.
        obstacles   : List of obstacles (each as a Shapely Polygon).
        num_rays    : Number of rays to cast over the primary FOV.
        secondary_view_range_factor: Factor to reduce view range for secondary FOV (default: 0.5).

    Returns:
        A Shapely Polygon representing the visible area.
    """
    points = []  # Will store the endpoints of each ray
    
    # Calculate the secondary view range
    secondary_range = view_range * secondary_view_range_factor
    
    # 1. Primary FOV rays (in the forward sector)
    primary_rays = num_rays
    rel_angles_primary = np.linspace(-fov/2, fov/2, primary_rays)
    
    for angle in rel_angles_primary:
        # Create a rotation matrix for the current ray angle
        rot_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                               [math.sin(angle), math.cos(angle)]])
        # Rotate the enemy's forward direction
        ray_dir = normalize(np.dot(rot_matrix, direction))
        # Compute the ideal end point of the ray (if unobstructed)
        end_point = pos + view_range * ray_dir
        # Create a line from the enemy's position to the end point
        ray = LineString([pos, end_point])
        nearest_dist = view_range  # Initialize with the full view range
        nearest_point = end_point  # If no obstacle, ray reaches end_point
        
        # Check for intersections with all obstacles
        for obs in obstacles:
            if hasattr(obs, 'intersection'):  # Shapely Polygon object
                inter = ray.intersection(obs)
                if not inter.is_empty:
                    # If the intersection is a point
                    if inter.geom_type == 'Point':
                        candidate = np.array(inter.coords[0])
                        d = np.linalg.norm(candidate - pos)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_point = candidate
                    # If multiple points, choose the closest one
                    elif inter.geom_type == 'MultiPoint':
                        for pt in inter:
                            candidate = np.array(pt.coords[0])
                            d = np.linalg.norm(candidate - pos)
                            if d < nearest_dist:
                                nearest_dist = d
                                nearest_point = candidate
                    # If the intersection is a line, take the first coordinate
                    elif inter.geom_type == 'LineString':
                        candidate = np.array(inter.coords[0])
                        d = np.linalg.norm(candidate - pos)
                        if d < nearest_dist:
                            nearest_dist = d
                            nearest_point = candidate
        
        # Append the endpoint for this ray
        points.append(nearest_point)
    
    # 2. Secondary FOV rays (complete 360° with reduced range)
    # Cast rays for the part outside the primary FOV
    secondary_rays = num_rays  # Number of rays for the secondary FOV
    # Calculate angles for the secondary FOV (outside the primary FOV sector)
    rel_angles_secondary = np.linspace(fov/2, 2*math.pi - fov/2, secondary_rays)
    
    for angle in rel_angles_secondary:
        # Create a rotation matrix for the current ray angle
        rot_matrix = np.array([[math.cos(angle), -math.sin(angle)],
                               [math.sin(angle), math.cos(angle)]])
        # Rotate the enemy's forward direction
        ray_dir = normalize(np.dot(rot_matrix, direction))
        # Compute the ideal end point of the ray with reduced range
        end_point = pos + secondary_range * ray_dir
        # Create a line from the enemy's position to the end point
        ray = LineString([pos, end_point])
        nearest_dist = secondary_range  # Initialize with the reduced view range
        nearest_point = end_point  # If no obstacle, ray reaches end_point
        
        # Check for intersections with all obstacles
        for obs in obstacles:
            if hasattr(obs, 'intersection'):  # Shapely Polygon object
                inter = ray.intersection(obs)
                if not inter.is_empty:
                    # Handle different intersection geometries
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
        
        # Append the endpoint for this ray
        points.append(nearest_point)
    
    # The FOV polygon is formed by the enemy's position and all the ray endpoints
    poly_coords = [pos] + points
    return Polygon(poly_coords)

# --------------------------
# Anticipatory Enemy FOV Prediction
# --------------------------
def predict_enemy_fov_union(enemy, obstacles, horizon_steps=8, pred_dt=0.5, buffer_m=0.2):
    """
    Predict the union of an enemy's FOV (sector + secondary ring) over a future time horizon.
    - Assumes constant velocity and heading over prediction horizon (lightweight and stable).
    - Accounts for occlusions by static obstacles when building each FOV polygon.
    - Returns a Shapely polygon (possibly MultiPolygon unified) buffered slightly for safety.
    """
    predicted_polys = []
    # Constant-velocity prediction
    for k in range(1, horizon_steps + 1):
        future_pos = enemy.pos + enemy.direction * enemy.speed * (pred_dt * k)
        vis_poly = compute_visibility_polygon_raycast(
            future_pos,
            enemy.direction,
            enemy.fov,
            enemy.view_range,
            obstacles,
            num_rays=50,
            secondary_view_range_factor=enemy.secondary_range_factor if hasattr(enemy, 'secondary_range_factor') else 0.3,
        )
        if hasattr(vis_poly, 'is_empty') and not vis_poly.is_empty:
            predicted_polys.append(vis_poly)
    if not predicted_polys:
        return None
    union_poly = unary_union(predicted_polys)
    if buffer_m and hasattr(union_poly, 'buffer'):
        try:
            union_poly = union_poly.buffer(buffer_m)
        except Exception:
            pass
    return union_poly

def build_anticipation_obstacles(enemies, static_obstacles, horizon_steps=8, pred_dt=0.5, buffer_m=0.2):
    """
    Build a list of predicted FOV union polygons for all enemies to be treated as temporary obstacles.
    """
    anticipation_polys = []
    for enemy in enemies:
        union_poly = predict_enemy_fov_union(enemy, static_obstacles, horizon_steps=horizon_steps, pred_dt=pred_dt, buffer_m=buffer_m)
        if union_poly is not None and hasattr(union_poly, 'is_empty') and not union_poly.is_empty:
            anticipation_polys.append(union_poly)
    return anticipation_polys

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
    angle_diff = math.acos(np.clip(np.dot(v_norm, enemy.direction), -1.0, 1.0))
    # Logistic functions for angle and distance.
    p_angle = 1.0 / (1.0 + math.exp(a * (angle_diff - angle_threshold)))
    p_dist  = 1.0 / (1.0 + math.exp(b * (d - dist_threshold)))
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
    Handles Shapely Polygon objects.
    """
    force = np.zeros(2)
    entity_point = Point(entity.pos)
    
    for obs in obstacles:
        if hasattr(obs, 'distance'):  # Shapely Polygon object
            # Find the nearest point on the obstacle to the entity
            if entity_point.distance(obs) < threshold:
                # If we're close to the obstacle, find the closest point on its boundary
                nearest_points = nearest_points_on_polygon(entity.pos, obs)
                if nearest_points:
                    nearest_point = np.array(nearest_points[0])
                    d = np.linalg.norm(entity.pos - nearest_point)
                    if d < threshold:
                        force += repulsive_coeff * (1.0/d - 1.0/threshold) * normalize(entity.pos - nearest_point)
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
        self.direction = normalize(self.goal - self.pos)  # Initialize direction towards goal

    def update(self, force, dt, smoothing=0.9, max_speed=1.0):
        """
        Updates the agent's velocity and position based on the applied force.
        Exponential smoothing is used to ensure smooth motion.
        """
        # Convert force to velocity commands
        linear_vel, angular_vel = force_to_velocity(force, self.direction)
        
        if USE_ROS:
            # In ROS mode, only publish velocity commands
            # Position will be updated from ROS odometry in the main loop
            ros_data.publish_velocity(linear_vel, angular_vel)
        else:
            # In simulation mode, update position internally
            # Convert velocity commands back to force for internal simulation
            force, self.direction = velocity_to_force(linear_vel, angular_vel, self.direction)
            
            self.velocity = smoothing * self.velocity + (1-smoothing) * force
            speed = np.linalg.norm(self.velocity)
            if speed > max_speed:
                self.velocity = (self.velocity/speed) * max_speed
            self.pos += self.velocity * dt

class EnemyAgent:
    # If you want to add a parameter to control the secondary FOV range factor:
    def __init__(self, pos, direction, fov=math.pi/3, view_range=5.0, speed=0.3, secondary_range_factor=0.3):
        self.pos = np.array(pos, dtype=float)
        self.direction = normalize(np.array(direction, dtype=float))
        self.fov = fov
        self.view_range = view_range
        self.speed = speed
        self.secondary_range_factor = secondary_range_factor

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
        c, s = math.cos(angle), math.sin(angle)
        rot = np.array([[c, -s], [s, c]])
        self.direction = normalize(np.dot(rot, self.direction))

    def can_see(self, point, obstacles):
        """
        Returns True if the enemy can see the given point.
        This uses the enhanced visibility polygon with both primary and secondary FOVs.
        """
        poly = self.get_visibility_polygon(obstacles)
        return poly.contains(Point(point))

        
    def get_visibility_polygon(self, obstacles):
        """
        Computes the enemy's current field of view polygon.
        """
        return compute_visibility_polygon_raycast(
            self.pos, 
            self.direction, 
            self.fov, 
            self.view_range, 
            obstacles,
            secondary_view_range_factor=self.secondary_range_factor
        )

# Function to create random convex polygons
def create_random_convex_polygon(sides=5):
    """
    Create a random convex polygon with at most 'sides' sides.
    """
    # Generate random points
    sides = random.randint(3, sides + 1)  # Between 3 and max sides
    center_x = random.uniform(scene_min, scene_max - 3)
    center_y = random.uniform(scene_min, scene_max - 3)
    
    # Generate points in a circle around the center
    radius = random.uniform(0.5, 2.0)
    angles = np.sort(np.random.uniform(0, 2*math.pi, sides))
    
    # Create coordinates
    coords = []
    for angle in angles:
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
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

agent = Agent(agent_start, agent_goal, max_speed=1.0)

# Initialize obstacles list
obstacles = []
obstacles_shapely = []

# For obstacle persistence: track how many consecutive steps each obstacle has been seen
obstacle_persistence_counter = defaultdict(int)
PERSISTENCE_THRESHOLD = 3  # Number of steps an obstacle must persist to be kept
persistent_obstacles = set()

# Add goal as a special obstacle
goal_poly = Polygon([(agent_goal[0]-0.5, agent_goal[1]-0.5), 
                     (agent_goal[0]+0.5, agent_goal[1]-0.5),
                     (agent_goal[0]+0.5, agent_goal[1]+0.5),
                     (agent_goal[0]-0.5, agent_goal[1]+0.5)])
obstacles_shapely.append(goal_poly)
bounds = goal_poly.bounds
centroid = get_polygon_centroid(goal_poly)
obstacles.append((goal_poly, bounds, centroid))

# Lists for storing simulation states for visualization.
agent_positions = [agent.pos.copy()]

# Store obstacles for each time step for animation
obstacles_per_step = []

# Ensure the first frame has the initial obstacles
import copy
obstacles_per_step.append(copy.deepcopy(obstacles_shapely))

past_agent_positions = []

# Globals for enemy animation data
enemy_positions_per_step = []
enemy_headings_per_step = []

stop_loop = False

def wait_for_key():
    global stop_loop
    input("Press Enter to stop the loop...\n")
    stop_loop = True


# --------------------------
# Simulation Loop
# --------------------------

# Main simulation loop
print("Starting robot-centric stealth simulation...")
print(f"Robot starting at: [0.0, 0.0] (robot frame)")
print(f"Goal at: [{ENVIRONMENT_MAX - 1}, {ENVIRONMENT_MAX - 1}] (robot frame)")
print(f"Simulation mode: {'ROS' if USE_ROS else 'Simulation'}")
print(f"Enemies enabled: {ENABLE_ENEMIES}")

# After robot position is known (after first odom), generate random enemy positions in grid space
# and spawn them in the real simulation space using subprocess and roslaunch. Wait 5 seconds after spawning.
ENEMY_NAMES = [f"husky{i+1}" for i in range(NUM_ENEMIES)]
enemy_spawn_positions = []
if USE_ROS:
    # Wait for first odom to get robot's world position
    while ros_data.get_odom() == [0.0, 0.0, 0.0]:
        time.sleep(0.1)
    robot_world_pos = ros_data.robot_start_pos if ros_data.robot_start_pos is not None else np.array([0.0, 0.0])
    print(f"Robot world start position: [{robot_world_pos[0]:.2f}, {robot_world_pos[1]:.2f}]")
    # Use fixed robot-frame spawns, then convert to world
    enemy_spawn_positions = []
    for i in range(min(NUM_ENEMIES, len(FIXED_ENEMY_SPAWNS_ROBOT))):
        spawn_pos = FIXED_ENEMY_SPAWNS_ROBOT[i]
        world_xy = ros_data.robot_to_world_frame(spawn_pos)
        print(f"Enemy {i+1}: Robot frame [{spawn_pos[0]:.2f}, {spawn_pos[1]:.2f}] -> World frame [{world_xy[0]:.2f}, {world_xy[1]:.2f}]")
        enemy_spawn_positions.append((spawn_pos[0], spawn_pos[1], world_xy[0], world_xy[1]))
    # Spawn each husky
    for idx, (x, y, wx, wy) in enumerate(enemy_spawn_positions):
        name = ENEMY_NAMES[idx]
        print(f"To spawn {name}, run:")
        print(f"source devel/setup.bash && roslaunch arl_unity_ros_ground husky.launch name:={name} x:={wx} y:={wy}")
        input('Press Enter after you have spawned all enemy huskies...')
    # Map: enemy index -> spawn world position
    enemy_spawn_world_pos = {idx: np.array([wx, wy]) for idx, (_, _, wx, wy) in enumerate(enemy_spawn_positions)}

threading.Thread(target=wait_for_key).start()
# Set up per-enemy ROS pub/sub after spawning via ros_data API
if USE_ROS:
    enemy_cmd_vel_pubs, enemy_odom_subs = ros_data.initialize_enemy_publishers_and_subscribers(ENEMY_NAMES)
    # Provide spawn world positions (for odom-delta localization)
    name_to_world = {ENEMY_NAMES[idx]: np.array([wx, wy]) for idx, (_, _, wx, wy) in enumerate(enemy_spawn_positions)}
    ros_data.set_enemy_spawn_world_positions(name_to_world)

while math.sqrt(abs(agent.pos[0] - agent_goal[0])**2 + abs(agent.pos[1] - agent_goal[1])**2) > 2 and not stop_loop:
    print(f"\n--- Simulation Step ---")
    print(f"Agent position: [{agent.pos[0]:.2f}, {agent.pos[1]:.2f}]")
    print(f"Distance to goal: {math.sqrt(abs(agent.pos[0] - agent_goal[0])**2 + abs(agent.pos[1] - agent_goal[1])**2):.2f}")
    
    # Update agent position from ROS odometry if in ROS mode
    if USE_ROS:
        odom = ros_data.get_odom()
        agent.pos = np.array([odom[0], odom[1]])
        agent.direction = np.array([math.cos(odom[2]), math.sin(odom[2])])
        print(f"Updated agent position from ROS: [{agent.pos[0]:.2f}, {agent.pos[1]:.2f}]")
    
    # Update obstacles based on mode
    if USE_ROS:
        pointcloud_data = ros_data.get_pointcloud()
        print(f"Point cloud data check: {len(pointcloud_data) if pointcloud_data else 0} points available")
        
        # Only process if we have new point cloud data
        if pointcloud_data and ros_data.has_new_pointcloud():
            print(f"Processing NEW point cloud with {len(pointcloud_data)} points...")
            
            # Dump point cloud data to file
            import json
            with open('pointcloud_dump.json', 'w') as f:
                json.dump(pointcloud_data, f, indent=2)
            print(f"Point cloud data dumped to pointcloud_dump.json")
            
            # Save point cloud visualization
            save_pointcloud_visualization(pointcloud_data, agent.pos, "pc.png")
            
            ros_data.process_pointcloud(pointcloud_data)
            ros_data.mark_pointcloud_processed()  # Mark as processed
            
            # Update obstacles in the simulation
            obstacles_shapely = ros_data.get_obstacles().copy()
            # Save obstacles visualization
            save_obstacles_visualization(obstacles_shapely, agent.pos, "obs.png")
            print(f"Updated obstacles from point cloud: {len(obstacles_shapely)} obstacles")
        elif pointcloud_data:
            print("Point cloud data available but no new data since last processing")
            # Use existing obstacles from previous processing
            obstacles_shapely = ros_data.get_obstacles().copy()
        else:
            print("No point cloud data available yet")
            obstacles_shapely = []
            # --- BEGIN OBSTACLE PERSISTENCE LOGIC ---
            # Update persistence counter for each detected obstacle
            new_persistent_obstacles = set()
            for obs in obstacles_shapely:
                found = False
                for existing_obs in persistent_obstacles:
                    if obs.equals(existing_obs):
                        obstacle_persistence_counter[existing_obs] += 1
                        new_persistent_obstacles.add(existing_obs)
                        found = True
                        break
                if not found:
                    obstacle_persistence_counter[obs] = 1
                    new_persistent_obstacles.add(obs)
            # Remove obstacles that haven't persisted long enough
            persistent_obstacles = set([obs for obs in new_persistent_obstacles if obstacle_persistence_counter[obs] >= PERSISTENCE_THRESHOLD])
            # Clean up counters for removed obstacles
            for obs in list(obstacle_persistence_counter.keys()):
                if obs not in persistent_obstacles:
                    del obstacle_persistence_counter[obs]
            obstacles_shapely = list(persistent_obstacles)
            # --- END OBSTACLE PERSISTENCE LOGIC ---
    else:
        obstacles_shapely = ros_data.get_obstacles().copy()
        obstacles_shapely.append(goal_poly)
        print(f"Using simulation obstacles: {len(obstacles_shapely)} obstacles")
        sample_pointcloud = ros_data.generate_sample_pointcloud()
        if sample_pointcloud:
            print("Processing sample point cloud for visualization...")
            save_pointcloud_visualization(sample_pointcloud, agent.pos, "pc.png")
            ros_data.process_pointcloud(sample_pointcloud)
            save_obstacles_visualization(obstacles_shapely, agent.pos, "obs.png")
            # --- BEGIN OBSTACLE PERSISTENCE LOGIC (SIM MODE) ---
            new_persistent_obstacles = set()
            for obs in obstacles_shapely:
                found = False
                for existing_obs in persistent_obstacles:
                    if obs.equals(existing_obs):
                        obstacle_persistence_counter[existing_obs] += 1
                        new_persistent_obstacles.add(existing_obs)
                        found = True
                        break
                if not found:
                    obstacle_persistence_counter[obs] = 1
                    new_persistent_obstacles.add(obs)
            persistent_obstacles = set([obs for obs in new_persistent_obstacles if obstacle_persistence_counter[obs] >= PERSISTENCE_THRESHOLD])
            for obs in list(obstacle_persistence_counter.keys()):
                if obs not in persistent_obstacles:
                    del obstacle_persistence_counter[obs]
            obstacles_shapely = list(persistent_obstacles)
            # --- END OBSTACLE PERSISTENCE LOGIC (SIM MODE) ---

    # --- Full Anticipatory Policy (from older_main.py, optimized for real-time) ---
    if USE_ANTICIPATORY_POLICY:
        good_obstacles = []
        
        # Create current enemies list for simulation
        current_enemies = []
        if USE_ROS:
            for idx_enemy in range(NUM_ENEMIES):
                pos = np.array(enemy_positions[idx_enemy])
                heading = enemy_headings[idx_enemy]
                direction = np.array([math.cos(heading), math.sin(heading)])
                current_enemies.append(EnemyAgent(pos, direction))
        else:
            # Use simulation enemies if available
            if hasattr(ros_data, 'sim_enemy_positions'):
                for idx in range(NUM_ENEMIES):
                    pos = ros_data.sim_enemy_positions[idx]
                    heading = ros_data.sim_enemy_headings[idx]
                    direction = np.array([math.cos(heading), math.sin(heading)])
                    current_enemies.append(EnemyAgent(pos, direction))
        
        #########################################################################################################
        # Simulating for all obstacles (comprehensive evaluation)
        #########################################################################################################
        for obstacle_index in range(len(obstacles)):
            if len(obstacles[obstacle_index]) < 3:
                continue  # Skip if obstacle doesn't have proper structure
                
            enemy_copies = []
            for enemy in current_enemies:
                enemy_copies.append(copy.deepcopy(enemy))
            agent_copy = copy.deepcopy(agent)

            # Get the centroid of the obstacle
            obstacle_centroid = obstacles[obstacle_index][2]
            
            # Don't go to the same one
            if math.sqrt(abs(agent_copy.pos[0] - obstacle_centroid[0])**2 + abs(agent_copy.pos[1] - obstacle_centroid[1])**2) < 2:
                print("same as curr obs")
                continue
            
            # Don't go against the goal
            if (abs(agent_copy.pos[0] - agent_goal[0])**2 + abs(agent_copy.pos[1] - agent_goal[1])**2) < (abs(agent_copy.pos[0] - obstacle_centroid[0])**2 + abs(agent_copy.pos[1] - obstacle_centroid[1])**2):
                print("bad obs")
                continue
            
            # Simulate movement to this obstacle (reduced steps for real-time performance)
            simulation_steps = 5  # Reduced from 10 for real-time performance
            simulation_dt = dt * 2  # Larger time step for faster simulation
            obstacle_is_safe = True
            
            for sim_step in range(simulation_steps):
                # Update enemy states during simulation
                for i, enemy in enumerate(enemy_copies):
                    enemy.update(simulation_dt, obstacles_shapely)
                    
                # Compute forces acting on the agent copy
                agent_copy.goal = np.array(obstacle_centroid, dtype=float)
                F_desired = desired_force(agent_copy, desired_speed=1.0) * 10.0
                # Anticipatory FOV union as temporary obstacles (do not occlude FOV raycast)
                predicted_fov_polys = build_anticipation_obstacles(enemy_copies, obstacles_shapely, horizon_steps=8, pred_dt=0.5, buffer_m=0.2)
                obstacles_with_anticipation = obstacles_shapely + (predicted_fov_polys if predicted_fov_polys else [])
                F_obs = obstacle_force(agent_copy, obstacles_with_anticipation, threshold=1.0, repulsive_coeff=100.0)

                F_enemy = np.zeros(2)
                for enemy in enemy_copies:
                    # Direct repulsion if inside enemy's current FOV
                    F_enemy += enemy_avoidance_force(agent_copy, enemy, weight=300.0, obstacles=obstacles_shapely)

                # Check if agent reached the subgoal
                if math.sqrt(abs(agent_copy.pos[0] - agent_copy.goal[0])**2 + abs(agent_copy.pos[1] - agent_copy.goal[1])**2) < 2:
                    good_obstacles.append(obstacle_index)
                    break
                    
                # Check if agent is detected by enemy (early termination)
                if not np.array_equal(F_enemy, np.zeros(2)):
                    obstacle_is_safe = False
                    break

                F_anticipatory = np.zeros(2)
                for enemy in enemy_copies:
                    # Anticipatory repulsion based on enemy prediction
                    F_anticipatory += anticipatory_enemy_avoidance_force(agent_copy, enemy, T=10.0, weight=1500.0)

                F_escape = fast_escape_force(agent_copy, enemy_copies, obstacles_shapely, weight=800.0)

                # Stuck detection during simulation
                is_stuck = is_agent_stuck(agent_copy, past_agent_positions, threshold=0.1, window=min(50, len(past_agent_positions)))
                
                if is_stuck:
                    escape_dir = find_escape_direction(agent_copy, obstacles_shapely, past_agent_positions)
                    F_escape_stuck = escape_dir * 25.0
                else:
                    F_escape_stuck = np.zeros(2)
                    
                F_total = F_desired + F_obs + F_escape_stuck
                
                # Update the agent copy
                agent_copy.update(F_total, simulation_dt, smoothing=0.9, max_speed=1.0)
                
                # Early termination if detected
                if not obstacle_is_safe:
                    break
    
        #########################################################################################################
        # Now performing actual movement based on best obstacle
        #########################################################################################################
        most_productive_obstacle = -1
        min_dist_to_final = float('inf')
        for i in range(len(good_obstacles)):
            obstacle_centroid = obstacles[good_obstacles[i]][2]
            x = float(obstacle_centroid[0]) - agent_goal[0]
            y = float(obstacle_centroid[1]) - agent_goal[1]
            dist_to_final = x**2 + y**2
            if dist_to_final < min_dist_to_final:
                min_dist_to_final = dist_to_final
                most_productive_obstacle = good_obstacles[i]
                
        if most_productive_obstacle == -1:
            print("should go into hiding here")
            agent.goal = agent_goal  # fallback to final goal
        else:
            obstacle_centroid = obstacles[most_productive_obstacle][2]
            print("most productive obstacle: ", obstacle_centroid, "currpos: ", agent.pos[0], agent.pos[1])
            agent.goal = np.array(obstacle_centroid, dtype=float)
        
        # Take multiple steps before re-evaluating (reduced for real-time)
        steps_before_reevaluation = 3  # Reduced from 10 for more responsive planning
        
        for step in range(steps_before_reevaluation):
            # Create current enemies for this step
            enemies = []
            if USE_ROS:
                for idx_enemy in range(NUM_ENEMIES):
                    pos = np.array(enemy_positions[idx_enemy])
                    heading = enemy_headings[idx_enemy]
                    direction = np.array([math.cos(heading), math.sin(heading)])
                    enemies.append(EnemyAgent(pos, direction))
            else:
                if hasattr(ros_data, 'sim_enemy_positions'):
                    for idx in range(NUM_ENEMIES):
                        pos = ros_data.sim_enemy_positions[idx]
                        heading = ros_data.sim_enemy_headings[idx]
                        direction = np.array([math.cos(heading), math.sin(heading)])
                        enemies.append(EnemyAgent(pos, direction))
            
            # Compute forces acting on the main agent
            F_desired = desired_force(agent, desired_speed=1.0) * 10.0
            # Build predicted FOV union polygons as temporary obstacles
            predicted_fov_polys = build_anticipation_obstacles(enemies, obstacles_shapely, horizon_steps=8, pred_dt=0.5, buffer_m=0.2)
            obstacles_with_anticipation = obstacles_shapely + (predicted_fov_polys if predicted_fov_polys else [])
            F_obs = obstacle_force(agent, obstacles_with_anticipation, threshold=1.0, repulsive_coeff=100.0)
            
            F_enemy = np.zeros(2)
            for enemy in enemies:
                F_enemy += enemy_avoidance_force(agent, enemy, weight=300.0, obstacles=obstacles_shapely)
                F_enemy += anticipatory_enemy_avoidance_force(agent, enemy, T=10.0, weight=1500.0)
            
            in_view = not np.array_equal(F_enemy, np.zeros(2))
            
            # Track past positions for stuck detection
            past_agent_positions.append(agent.pos.copy())
            if len(past_agent_positions) > 100:
                past_agent_positions.pop(0)
            
            # Check if agent is stuck
            is_stuck = is_agent_stuck(agent, past_agent_positions, threshold=0.1, window=100)
            
            if is_stuck:
                escape_dir = find_escape_direction(agent, obstacles_shapely, past_agent_positions)
                F_escape_stuck = escape_dir * 25.0
            else:
                F_escape_stuck = np.zeros(2)
            
            F_total = F_desired + F_obs + F_enemy + F_escape_stuck
            
            # Update the agent
            agent.update(F_total, dt, smoothing=0.9, max_speed=(1.0 if not in_view else 2.0))
            agent_positions.append(agent.pos.copy())
            
            # Check if we've reached the current intermediate goal
            if most_productive_obstacle != -1:
                obstacle_centroid = obstacles[most_productive_obstacle][2]
                if math.sqrt(abs(agent.pos[0] - obstacle_centroid[0])**2 + abs(agent.pos[1] - obstacle_centroid[1])**2) < 2:
                    print("Reached subgoal")
                    break
            
            # Check if we're close enough to the final goal
            if math.sqrt(abs(agent.pos[0] - agent_goal[0])**2 + abs(agent.pos[1] - agent_goal[1])**2) <= 2:
                print("Close enough to final goal")
                break
        
        # At the end of each step, store a deep copy of the current obstacles for animation
        obstacles_per_step.append(copy.deepcopy(obstacles_shapely))
        continue  # Skip the rest of the loop (already did steps)
    # --- End Anticipatory Policy ---

    # --- Simple Policy (original) ---
    F_desired = desired_force(agent, desired_speed=1.0) * 10.0
    # Even in simple policy, repel from predicted future enemy FOVs if enemies are present
    simple_policy_enemies = []
    if USE_ROS:
        for idx_enemy in range(NUM_ENEMIES):
            pos = np.array(enemy_positions[idx_enemy])
            heading = enemy_headings[idx_enemy]
            direction = np.array([math.cos(heading), math.sin(heading)])
            simple_policy_enemies.append(EnemyAgent(pos, direction))
    else:
        if hasattr(ros_data, 'sim_enemy_positions'):
            for idx in range(NUM_ENEMIES):
                pos = ros_data.sim_enemy_positions[idx]
                heading = ros_data.sim_enemy_headings[idx]
                direction = np.array([math.cos(heading), math.sin(heading)])
                simple_policy_enemies.append(EnemyAgent(pos, direction))
    predicted_fov_polys_simple = build_anticipation_obstacles(simple_policy_enemies, obstacles_shapely, horizon_steps=8, pred_dt=0.5, buffer_m=0.2)
    obstacles_with_anticipation_simple = obstacles_shapely + (predicted_fov_polys_simple if predicted_fov_polys_simple else [])
    F_obs = obstacle_force(agent, obstacles_with_anticipation_simple, threshold=1.0, repulsive_coeff=100.0)
    past_agent_positions.append(agent.pos.copy())
    if len(past_agent_positions) > 100:
        past_agent_positions.pop(0)
    is_stuck = is_agent_stuck(agent, past_agent_positions, threshold=0.1, window=100)
    if is_stuck:
        escape_dir = find_escape_direction(agent, obstacles_shapely, past_agent_positions)
        F_escape_stuck = escape_dir * 25.0
        print("Agent stuck, applying escape force")
    else:
        F_escape_stuck = np.zeros(2)
    F_total = F_desired + F_obs + F_escape_stuck
    agent.update(F_total, dt, smoothing=0.9, max_speed=1.0)
    agent_positions.append(agent.pos.copy())
    print(f"Agent moved to: [{agent.pos[0]:.2f}, {agent.pos[1]:.2f}]")
    if math.sqrt(abs(agent.pos[0] - agent_goal[0])**2 + abs(agent.pos[1] - agent_goal[1])**2) <= 2:
        print("Close enough to final goal!")
        break
    obstacles_per_step.append(copy.deepcopy(obstacles_shapely))

    # In the main loop, after agent update, update each enemy with proper obstacle avoidance
    if USE_ROS:
        if len(enemy_positions_per_step) == 0:
            enemy_positions_per_step = []
            enemy_headings_per_step = []
        enemy_positions_per_step.append([enemy_positions[i][:] for i in range(NUM_ENEMIES)])
        enemy_headings_per_step.append([enemy_headings[i] for i in range(NUM_ENEMIES)])
        
        # Create proper EnemyAgent objects for obstacle avoidance
        enemy_agents = []
        for idx in range(NUM_ENEMIES):
            pos = np.array(enemy_positions[idx])
            heading = enemy_headings[idx]
            direction = np.array([math.cos(heading), math.sin(heading)])
            enemy_agents.append(EnemyAgent(pos, direction))
        
        # Update each enemy with obstacle avoidance
        for idx, enemy in enumerate(enemy_agents):
            # Apply obstacle avoidance force (stronger than in older_main.py for safety)
            F_obs = obstacle_force(enemy, obstacles_shapely, threshold=1.0, repulsive_coeff=200.0)
            
            # Update enemy direction based on obstacle force
            new_direction = normalize(enemy.direction + 0.5 * F_obs)  # Stronger obstacle influence
            
            # Calculate desired movement
            desired_velocity = new_direction * 0.3  # Base speed
            
            # Add some randomness for natural movement
            if random.random() < 0.1:  # 10% chance for random direction change
                random_angle = random.uniform(-0.2, 0.2)
                c, s = math.cos(random_angle), math.sin(random_angle)
                rot = np.array([[c, -s], [s, c]])
                new_direction = normalize(np.dot(rot, new_direction))
            
            # Convert to linear and angular velocities
            linear_vel = np.linalg.norm(desired_velocity)
            
            # Calculate angular velocity to turn towards new direction
            current_heading = enemy_headings[idx]
            desired_heading = math.atan2(new_direction[1], new_direction[0])
            heading_diff = desired_heading - current_heading
            
            # Normalize heading difference to [-pi, pi]
            while heading_diff > math.pi:
                heading_diff -= 2 * math.pi
            while heading_diff < -math.pi:
                heading_diff += 2 * math.pi
            
            # Limit angular velocity
            max_angular_vel = 0.5
            angular_vel = np.clip(heading_diff / dt, -max_angular_vel, max_angular_vel)
            
            # Add small random angular velocity for natural movement
            if random.random() < 0.05:
                angular_vel += random.uniform(-0.1, 0.1)
            
            # Publish velocity commands
            twist = Twist()
            twist.linear.x = linear_vel
            twist.angular.z = angular_vel
            enemy_cmd_vel_pubs[idx].publish(twist)
    else:
        if len(enemy_positions_per_step) == 0:
            enemy_positions_per_step = []
            enemy_headings_per_step = []
        
        # Initialize simulation enemies with proper obstacle avoidance spawn positions
        if not hasattr(ros_data, 'sim_enemy_agents'):
            ros_data.sim_enemy_agents = []
            for i in range(NUM_ENEMIES):
                # Use similar obstacle avoidance spawn logic as ROS mode
                max_attempts = 50
                attempts = 0
                while attempts < max_attempts:
                    x = random.uniform(scene_min + 2, scene_max - 2)
                    y = random.uniform(scene_min + 2, scene_max - 2)
                    spawn_pos = np.array([x, y])
                    
                    # Avoid spawning such that enemy FOV overlaps with agent position
                    # Enemy has view_range=5.0, so ensure distance > view_range + safety margin
                    min_distance = 5.0 + 1.0  # FOV radius + 1.0 safety margin
                    if np.linalg.norm(spawn_pos - agent.pos) <= min_distance:
                        attempts += 1
                        continue
                    
                    # Check collision with obstacles
                    spawn_point = Point(spawn_pos)
                    collision = False
                    for obs in obstacles_shapely:
                        if hasattr(obs, 'contains') and obs.contains(spawn_point):
                            collision = True
                            break
                        if hasattr(obs, 'distance') and spawn_point.distance(obs) < 1.0:
                            collision = True
                            break
                    
                    if not collision:
                        break
                    attempts += 1
                
                if attempts >= max_attempts:
                    # Fallback position (at least 6 units away to avoid FOV overlap)
                    spawn_pos = np.array([scene_min + 7 + i * 3, scene_min + 7 + i * 3])
                
                # Create EnemyAgent with random initial direction
                initial_direction = np.array([math.cos(random.uniform(0, 2*math.pi)), 
                                           math.sin(random.uniform(0, 2*math.pi))])
                enemy_agent = EnemyAgent(spawn_pos, initial_direction, speed=0.3)
                ros_data.sim_enemy_agents.append(enemy_agent)
        
        # Update each enemy using proper EnemyAgent.update() method
        for idx, enemy in enumerate(ros_data.sim_enemy_agents):
            # Use the same update method as in older_main.py
            enemy.update(dt, obstacles_shapely)
            
            # Keep enemies within scene boundaries (bounce off walls)
            if enemy.pos[0] < scene_min or enemy.pos[0] > scene_max:
                enemy.direction[0] = -enemy.direction[0]
                enemy.pos[0] = np.clip(enemy.pos[0], scene_min, scene_max)
            if enemy.pos[1] < scene_min or enemy.pos[1] > scene_max:
                enemy.direction[1] = -enemy.direction[1]
                enemy.pos[1] = np.clip(enemy.pos[1], scene_min, scene_max)
        
        # Store positions and headings for visualization
        enemy_positions_per_step.append([enemy.pos.copy() for enemy in ros_data.sim_enemy_agents])
        enemy_headings_per_step.append([math.atan2(enemy.direction[1], enemy.direction[0]) for enemy in ros_data.sim_enemy_agents])
        
        # Update ros_data attributes for compatibility with other parts of code
        ros_data.sim_enemy_positions = {i: ros_data.sim_enemy_agents[i].pos for i in range(NUM_ENEMIES)}
        ros_data.sim_enemy_headings = {i: math.atan2(ros_data.sim_enemy_agents[i].direction[1], ros_data.sim_enemy_agents[i].direction[0]) for i in range(NUM_ENEMIES)}

# Convert stored positions to numpy arrays for visualization.
agent_positions = np.array(agent_positions)
print(f"\nSimulation completed!")
print(f"Total agent positions recorded: {len(agent_positions)}")
print(f"Final agent position: [{agent_positions[-1][0]:.2f}, {agent_positions[-1][1]:.2f}]")
        
# --------------------------
# Visualization Setup
# --------------------------
print("Setting up visualization...")
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(ENVIRONMENT_MIN-2, ENVIRONMENT_MAX+2)
ax.set_ylim(ENVIRONMENT_MIN-2, ENVIRONMENT_MAX+2)
ax.set_aspect('equal')
ax.set_title(f"Robot-Centric Navigation with Point Cloud Obstacles\nEnvironment: ({ENVIRONMENT_MIN}, {ENVIRONMENT_MIN}) to ({ENVIRONMENT_MAX}, {ENVIRONMENT_MAX})")

# Remove static obstacle drawing here; will be handled in animation
# Initialize the agent's trajectory line and its marker.
line, = ax.plot([], [], 'b-', lw=2, label='Agent Path')
agent_dot, = ax.plot([], [], 'bo', ms=8, label='Agent')

# Prepare a list to hold obstacle patches for animation
obstacle_patches = []

# Add legend
ax.legend()

# --------------------------
# Animation Functions
# --------------------------
def init():
    line.set_data([], [])
    agent_dot.set_data([], [])
    # Remove any existing obstacle patches
    for patch in obstacle_patches:
        patch.remove()
    obstacle_patches.clear()
    return line, agent_dot

def animate(i):
    # Update agent's trajectory.
    if i < len(agent_positions):
        # Fix slicing for agent_positions (convert to numpy array for correct indexing)
        pos_arr = np.array(agent_positions)
        line.set_data(pos_arr[:i+1, 0], pos_arr[:i+1, 1])
        agent_dot.set_data([pos_arr[i, 0]], [pos_arr[i, 1]])
    # Remove previous obstacle patches
    for patch in obstacle_patches:
        patch.remove()
    obstacle_patches.clear()
    # Draw obstacles for this frame
    if i < len(obstacles_per_step) and len(obstacles_per_step[i]) > 0:
        for obs in obstacles_per_step[i]:
            if hasattr(obs, 'exterior'):
                coords = np.array(obs.exterior.coords)
                # Draw goal in green, others in gray
                if obs.equals(goal_poly):
                    patch = MplPolygon(coords, closed=True, color='green', alpha=0.7, label='Goal')
                else:
                    patch = MplPolygon(coords, closed=True, color='gray', alpha=0.5)
                ax.add_patch(patch)
                obstacle_patches.append(patch)
    else:
        # Always show the goal polygon if nothing else
        coords = np.array(goal_poly.exterior.coords)
        patch = MplPolygon(coords, closed=True, color='green', alpha=0.7, label='Goal')
        ax.add_patch(patch)
        obstacle_patches.append(patch)

    # In animation, draw enemies and their FOVs each frame
    if USE_ROS:
        # Use live enemy positions/headings if available
        for idx in range(NUM_ENEMIES):
            if (idx in enemy_positions) and (idx in enemy_headings):
                pos = enemy_positions[idx]
                heading = enemy_headings[idx]
                if not (np.allclose(pos[0], 0.0, atol=1e-6) and np.allclose(pos[1], 0.0, atol=1e-6)):
                    ax.scatter(pos[0], pos[1], c='orange', s=80, marker='s', label=f'Enemy {idx+1}' if i == 0 else "")
                    direction = np.array([math.cos(heading), math.sin(heading)])
                    enemy = EnemyAgent(np.array(pos), direction)
                    fov_poly = enemy.get_visibility_polygon(obstacles_per_step[i] if i < len(obstacles_per_step) else [])
                    if hasattr(fov_poly, 'exterior'):
                        coords = np.array(fov_poly.exterior.coords)
                        patch = MplPolygon(coords, closed=True, color='orange', alpha=0.2)
                        ax.add_patch(patch)
                        obstacle_patches.append(patch)
    else:
        if 'enemy_positions_per_step' in globals() and len(enemy_positions_per_step) > 0:
            for idx in range(NUM_ENEMIES):
                if i < len(enemy_positions_per_step):
                    pos = enemy_positions_per_step[i][idx]
                    heading = enemy_headings_per_step[i][idx]
                    ax.scatter(pos[0], pos[1], c='orange', s=80, marker='s', label=f'Enemy {idx+1}' if i == 0 else "")
                    # Draw FOV polygon
                    direction = np.array([math.cos(heading), math.sin(heading)])
                    enemy = EnemyAgent(pos, direction)
                    fov_poly = enemy.get_visibility_polygon(obstacles_per_step[i] if i < len(obstacles_per_step) else [])
                    if hasattr(fov_poly, 'exterior'):
                        coords = np.array(fov_poly.exterior.coords)
                        patch = MplPolygon(coords, closed=True, color='orange', alpha=0.2)
                        ax.add_patch(patch)
                        obstacle_patches.append(patch)
    return (line, agent_dot, *obstacle_patches)

# Create the animation.
print("Creating animation...")
ani = animation.FuncAnimation(fig, animate, frames=len(agent_positions),
                              interval=50, init_func=init)

# Save the animation as an MP4 video
print("Saving animation as MP4...")
ani.save('animation.mp4', writer='ffmpeg', fps=20)
print("Animation saved as animation.mp4")
plt.close(fig)  # Close the figure after saving
print("Simulation and visualization complete!")