import numpy as np
import matplotlib.pyplot as plt

# Define grid
x = np.linspace(-10, 15, 150)
y = np.linspace(-10, 15, 150)
X, Y = np.meshgrid(x, y)

# Goal position (cone center)
x_goal, y_goal = 5, 5

# Cone function (sloping down to goal)
lambda_cone = 1.0
cone = lambda_cone * np.sqrt((X - x_goal)**2 + (Y - y_goal)**2)

# Obstacles as Gaussian bumps
obstacles = np.zeros_like(X)
obstacle_centers = [(-3, 2), (2, -4), (4, 3)]
obstacle_heights = [10, 8, 12]
sigma = 1.5

for (ox, oy), height in zip(obstacle_centers, obstacle_heights):
    obstacles += height * np.exp(-((X - ox)**2 + (Y - oy)**2) / (2 * sigma**2))

# Enemy FOVs
enemy_data = [(0, 0, 0), (10, 10, 135), (5, -2, 90), (8, 3, -45)]
fov_radius = 3.0
fov_angle = 30
plateau_height = 15  # Keep the plateau effect
transition_smoothness = 1.5  # Controls the blending of FOV into terrain

fov_sectors = np.zeros_like(X)

def smooth_plateau(px, py, ex, ey, facing_angle, fov_angle, radius, height, smoothness):
    dx, dy = px - ex, py - ey
    distance = np.sqrt(dx**2 + dy**2)
    point_angle = np.degrees(np.arctan2(dy, dx))
    angle_diff = (point_angle - facing_angle + 180) % 360 - 180

    # Check if the point is inside the sector
    inside_fov = (abs(angle_diff) <= fov_angle / 2) and (distance < radius)

    # Create smooth transition using a sigmoid function near the edges
    distance_blend = 1 / (1 + np.exp((distance - radius) / smoothness))
    angle_blend = 1 / (1 + np.exp((abs(angle_diff) - fov_angle / 2) / smoothness))

    # Plateau inside the FOV, with smooth edges
    return height * inside_fov + height * distance_blend * angle_blend * (not inside_fov)

for ex, ey, facing_angle in enemy_data:
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fov_sectors[i, j] += smooth_plateau(X[i, j], Y[i, j], ex, ey, facing_angle, fov_angle, fov_radius, plateau_height, transition_smoothness)

# Final terrain
terrain = cone + obstacles + fov_sectors


# Plot 3D Surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, terrain, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Cost (Height)')
ax.set_title('Gradient Descent Terrain with Obstacles & Smooth Enemy FOV Sectors')

plt.show()

# Compute gradient using finite differences
dx = np.gradient(terrain, axis=1)
dy = np.gradient(terrain, axis=0)

# Gradient descent navigation
start_x, start_y = -5, -5
alpha = 0.2  # Step size
max_steps = 1000

agent_x, agent_y = start_x, start_y
path_x, path_y = [agent_x], [agent_y]

for _ in range(max_steps):
    # Find closest grid point
    i = np.argmin(np.abs(x - agent_x))
    j = np.argmin(np.abs(y - agent_y))
    
    # Compute gradient at current position
    grad_x, grad_y = dx[j, i], dy[j, i]

    # Adjust step size dynamically based on terrain steepness
    step_size = alpha / (1 + 0.1 * np.linalg.norm([grad_x, grad_y]))
    
    # Gradient descent update
    agent_x -= step_size * grad_x
    agent_y -= step_size * grad_y

    # Store path
    path_x.append(agent_x)
    path_y.append(agent_y)

    # Stop if close to goal
    if np.linalg.norm([agent_x - x_goal, agent_y - y_goal]) < 0.1:
        break

# Plot terrain with path
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, terrain, levels=50, cmap='viridis')
plt.colorbar(contour)


# Plot enemy positions and directions
plt.scatter([x_goal], [y_goal], color='red', marker='X', s=100, label="Goal")
plt.scatter(*zip(*obstacle_centers), color='white', marker='o', s=100, label="Obstacles")
for ex, ey, _ in enemy_data:
    plt.scatter(ex, ey, color='red', marker='o', s=100)  # Enemy positions

# Plot agent path
plt.plot(path_x, path_y, color='cyan', linestyle='-', marker='o', markersize=3, label="Agent Path")
plt.scatter([start_x], [start_y], color='blue', marker='o', s=100, label="Start")

plt.legend()
plt.title("Gradient Descent Pathfinding with Smooth Plateau FOV")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()