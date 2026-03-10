import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define grid
x = np.linspace(-10, 15, 150)
y = np.linspace(-10, 15, 150)
X, Y = np.meshgrid(x, y)

# Goal position (cone center)
x_goal, y_goal = 5, 5

# Cone function (sloping down to goal)
lambda_cone = 1.0  # Steepness of cone
cone = lambda_cone * np.sqrt((X - x_goal)**2 + (Y - y_goal)**2)

# Define obstacles as Gaussian bumps
obstacles = np.zeros_like(X)
obstacle_centers = [(-3, 2), (2, -4), (4, 3)]
obstacle_heights = [10, 8, 12]
sigma = 1.5  # Spread of obstacles

for (ox, oy), height in zip(obstacle_centers, obstacle_heights):
    obstacles += height * np.exp(-((X - ox)**2 + (Y - oy)**2) / (2 * sigma**2))

# Define enemy FOVs as smooth sigmoid transitions
enemy_data = [
    (0, 0, 0),     # (x, y, facing angle in degrees)
    (10, 10, 135),
    (5, -2, 90),
    (8, 3, -45)
]

fov_radius = 3.0  # FOV range
fov_angle = 30  # FOV spread in degrees
plateau_height = 15  # Max penalty in FOV
epsilon = 0.5  # Controls the smoothness of transition

fov_sectors = np.zeros_like(X)

# Sigmoid function for smooth transition
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to check if a point is inside an enemy's FOV sector (smooth version)
def smooth_fov(px, py, ex, ey, facing_angle, fov_angle, radius, epsilon):
    dx, dy = px - ex, py - ey
    distance = np.sqrt(dx**2 + dy**2)
    
    # Compute angle difference
    point_angle = np.degrees(np.arctan2(dy, dx))
    angle_diff = (point_angle - facing_angle + 180) % 360 - 180  # Normalize to [-180, 180]

    # Sigmoid-based penalty (smooth transition)
    fov_penalty = plateau_height * sigmoid((radius - distance) / epsilon) * sigmoid((fov_angle / 2 - abs(angle_diff)) / epsilon)
    
    return fov_penalty

# Apply the smooth sector function to the entire grid
for ex, ey, facing_angle in enemy_data:
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            fov_sectors[i, j] += smooth_fov(X[i, j], Y[i, j], ex, ey, facing_angle, fov_angle, fov_radius, epsilon)

# Final terrain = Cone + Obstacles + Smooth Enemy FOV Sectors
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

# Optional: 2D Contour Plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, terrain, levels=50, cmap='viridis')
plt.colorbar(contour)

# Plot enemy positions and directions
plt.scatter([x_goal], [y_goal], color='red', marker='X', s=100, label="Goal")
plt.scatter(*zip(*obstacle_centers), color='white', marker='o', s=100, label="Obstacles")
for ex, ey, facing_angle in enemy_data:
    plt.scatter(ex, ey, color='red', marker='o', s=100)  # Enemy positions
plt.legend()
plt.title("Contour Map of Terrain with Smooth Enemy FOV Sectors")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

