import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib import animation
from IPython.display import Video
import copy
import threading
import math

# Shapely is used for geometric computations (polygons, intersections, etc.)
from shapely.geometry import Polygon, Point, box, LineString
from shapely.ops import unary_union

# --------------------------
# Scene Setup & Utility Functions
# --------------------------

# Define the simulation boundaries.
scene_min, scene_max = -5, 15

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
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
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
    rel_angles_secondary = np.linspace(fov/2, 2*np.pi - fov/2, secondary_rays)
    
    for angle in rel_angles_secondary:
        # Create a rotation matrix for the current ray angle
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
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
    Modified to work with convex polygons.
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
# Simulation Setup
# --------------------------
dt = 0.25                     # Time step (seconds)
sim_time = 60.0               # Total simulation time (seconds)
steps = int(sim_time / dt)    # Total number of simulation steps

# Main agent starts at bottom-left and has a goal at top-right.
agent_start = np.array([scene_min, scene_min])
agent_goal = np.array([scene_max, scene_max])
agent = Agent(agent_start, agent_goal, max_speed=1.0)

# Create enemy agents with random positions and directions.
num_enemies = 5
enemies = []
enemies_data = []
for _ in range(num_enemies):
    pos = np.random.uniform(scene_min + 2, scene_max, size=2)
    angle = np.random.uniform(0, 2*np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)])
    fov = np.pi/3 + np.random.uniform(-0.1, 0.1)
    enemies_data.append(EnemyAgent(pos, direction, fov, speed=0.3))
    enemies.append(EnemyAgent(pos, direction, fov, speed=0.3))

# Create obstacles as convex polygons
num_obstacles = 6
obstacles = []
obstacles_shapely = []

past_agent_positions = []

for _ in range(num_obstacles):
    poly = create_random_convex_polygon(sides=5)
    obstacles_shapely.append(poly)
    
    # Store polygon's bounds and centroid for later use
    bounds = poly.bounds  # Returns (minx, miny, maxx, maxy)
    centroid = get_polygon_centroid(poly)
    
    # Store the polygon, its bounds and centroid
    obstacles.append((poly, bounds, centroid))

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
enemy_positions = [[] for _ in range(num_enemies)]
enemy_directions = [[] for _ in range(num_enemies)]

stop_loop = False

def wait_for_key():
    global stop_loop
    input("Press Enter to stop the loop...\n")
    stop_loop = True

# Start the key listener in a separate thread
threading.Thread(target=wait_for_key).start()

# --------------------------
# Simulation Loop
# --------------------------

# Main simulation loop
while math.sqrt(abs(agent.pos[0] - agent_goal[0])**2 + abs(agent.pos[1] - agent_goal[1])**2) > 2 and not stop_loop:
    good_obstacles = []
    unproductive_safe_obstacles = []
    prev_enemy_pos = []

    #########################################################################################################
    # Simulating for all obstacles
    #########################################################################################################
    for obstacle_index in range(len(obstacles)):
        enemy_copies = []
        for enemy in enemies:
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
        
        while(True):
            # Update enemy states.
            for i, enemy in enumerate(enemy_copies):
                enemy.update(dt, obstacles_shapely)
                
            # Compute forces acting on the main agent_copy.
            agent_copy.goal = np.array(obstacle_centroid, dtype=float)
            F_desired = desired_force(agent_copy, desired_speed=1.0) * 10.0            # Strong pull toward the goal.
            F_obs = obstacle_force(agent_copy, obstacles_shapely, threshold=1.0, repulsive_coeff=100.0)  # Repulsion from obstacles.

            F_enemy = np.zeros(2)
            for enemy in enemy_copies:
                # Direct repulsion if inside enemy's current FOV.
                F_enemy += enemy_avoidance_force(agent_copy, enemy, weight=300.0, obstacles=obstacles_shapely)

            in_view = False
            if math.sqrt(abs(agent_copy.pos[0] - agent_copy.goal[0])**2 + abs(agent_copy.pos[1] - agent_copy.goal[1])**2) < 2:
                good_obstacles.append(obstacle_index)
                break
            if not np.array_equal(F_enemy, np.zeros(2)):
                in_view = True
                break

            F_anticipatory = np.zeros(2)
            for enemy in enemy_copies:
                # Anticipatory repulsion based on enemy prediction (10 sec ahead).
                F_anticipatory += anticipatory_enemy_avoidance_force(agent_copy, enemy, T=10.0, weight=1500.0)

            F_escape = fast_escape_force(agent_copy, enemy_copies, obstacles_shapely, weight=800.0)

            # Track past positions for stuck detection
            past_agent_positions.append(agent.pos.copy())
            if len(past_agent_positions) > 100:  # Keep only the last 20 positions
                past_agent_positions.pop(0)
            
            # Check if agent is stuck
            is_stuck = is_agent_stuck(agent, past_agent_positions, threshold=0.1, window=100)
            
            if is_stuck:
                escape_dir = find_escape_direction(agent, obstacles_shapely, past_agent_positions)
                F_escape_stuck = escape_dir * 25.0  # Strong escape force
                print("Agent stuck, applying escape force.")
            else:
                F_escape_stuck = np.zeros(2)
                
            F_total = F_desired + F_obs + F_escape_stuck

            # Update the agent copy
            agent_copy.update(F_total, dt, smoothing=0.9, max_speed=(1.0 if not in_view else 2.0))
    
    #########################################################################################################
    # Now performing actual movement
    #########################################################################################################
    most_productive_obstacle = -1
    min_dist_to_final = 10000
    for i in range(len(good_obstacles)):
        obstacle_centroid = obstacles[good_obstacles[i]][2]
        x = float(obstacle_centroid[0]) - agent_goal[0] # can change to prioritize dist to final or curr
        y = float(obstacle_centroid[1]) - agent_goal[1] # can change to prioritize dist to final or curr
        if (x**2 + y**2) < min_dist_to_final:
            min_dist_to_final = (x**2 + y**2)
            most_productive_obstacle = good_obstacles[i]
            
    if (most_productive_obstacle == -1):
        print("should go into hiding here")
    else:
        obstacle_centroid = obstacles[most_productive_obstacle][2]
        print("most productive obstacle: ", obstacle_centroid, "currpos: ", agent.pos[0], agent.pos[1])
    
    # Set the agent's goal based on the most productive obstacle
    if (most_productive_obstacle != -1):
        obstacle_centroid = obstacles[most_productive_obstacle][2]
        agent.goal = np.array(obstacle_centroid, dtype=float)
    
    # take multiple steps before re-evaluating
    steps_before_reevaluation = 10
    
    for step in range(steps_before_reevaluation):
        # Update enemy states
        for i, enemy in enumerate(enemies):
            enemy.update(dt, obstacles_shapely)
            enemy_positions[i].append(enemy.pos.copy())
            enemy_directions[i].append(enemy.direction.copy())
        
        # Compute forces acting on the main agent
        F_desired = desired_force(agent, desired_speed=1.0) * 10.0
        F_obs = obstacle_force(agent, obstacles_shapely, threshold=1.0, repulsive_coeff=100.0)
        
        F_enemy = np.zeros(2)
        for enemy in enemies:
            F_enemy += enemy_avoidance_force(agent, enemy, weight=300.0, obstacles=obstacles_shapely)
        
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
            
    # Update agent_start for the next iteration
    agent_start = np.array(agent.pos.copy())
    
# Convert stored positions to numpy arrays for visualization.
agent_positions = np.array(agent_positions)
# Extend enemy state lists to match the length of agent state list.
if enemy_positions and len(enemy_positions[0]) < len(agent_positions):
    for i in range(num_enemies):
        enemy_positions[i].append(enemy_positions[i][-1])
        enemy_directions[i].append(enemy_directions[i][-1])
        
# --------------------------
# Visualization Setup
# --------------------------
fig, ax = plt.subplots(figsize=(10,10))
ax.set_xlim(scene_min-2, scene_max+2)
ax.set_ylim(scene_min-2, scene_max+2)
ax.set_aspect('equal')
ax.set_title("Stealth Simulation with Enhanced Anticipation")

# Draw obstacles as polygons
for obs_data in obstacles:
    poly, _, _ = obs_data
    coords = np.array(poly.exterior.coords)
    ax.add_patch(MplPolygon(coords, closed=True, color='gray', alpha=0.5))

# Initialize enemy markers and their visibility polygons.
enemy_dots = []
enemy_vis_patches = []
for enemy in enemies:
    dot, = ax.plot(enemy.pos[0], enemy.pos[1], 'ro')
    enemy_dots.append(dot)
    vis_poly = enemy.get_visibility_polygon(obstacles_shapely)
    patch = MplPolygon(np.array(vis_poly.exterior.coords), closed=True, color='red', alpha=0.2)
    ax.add_patch(patch)
    enemy_vis_patches.append(patch)

# Initialize the agent's trajectory line and its marker.
line, = ax.plot([], [], 'b-', lw=2)
agent_dot, = ax.plot([], [], 'bo', ms=8)

# --------------------------
# Animation Functions
# --------------------------
def init():
    line.set_data([], [])
    agent_dot.set_data([], [])
    return line, agent_dot, *enemy_dots, *enemy_vis_patches

def animate(i):
    # Update agent's trajectory.
    line.set_data(agent_positions[:i, 0], agent_positions[:i, 1])
    agent_dot.set_data([agent_positions[i, 0]], [agent_positions[i, 1]])
    # Update enemy positions and their visibility polygons.
    for j in range(num_enemies):
        pos = enemy_positions[j][i]
        enemy_dots[j].set_data([pos[0]], [pos[1]])
        dvec = enemy_directions[j][i]
        vis_poly = compute_visibility_polygon_raycast(pos, dvec, enemies[j].fov, enemies[j].view_range, obstacles_shapely)
        enemy_vis_patches[j].set_xy(np.array(vis_poly.exterior.coords))
    return line, agent_dot, *enemy_dots, *enemy_vis_patches

# Create the animation.
ani = animation.FuncAnimation(fig, animate, frames=len(agent_positions),
                              interval=50, blit=True, init_func=init)
plt.close()

# Save the animation as an MP4 video and display it.
ani.save('animation.mp4', writer='ffmpeg', fps=20)
Video('animation.mp4', embed=True)