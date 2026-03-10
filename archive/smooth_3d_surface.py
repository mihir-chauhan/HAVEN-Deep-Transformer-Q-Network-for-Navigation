import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

height = 400
x_goal = 15
y_goal = 15

def moat_sigmoid(x, a=10, b=1.5):
    return sigmoid(x, a)
    term1 = 1 / (b + np.exp((x + 0.5) * a))
    term2 = (1 / b + 1) / (1 + np.exp(-a * x))
    term3 = -1 / b
    return term1 + term2 + term3
    
def sigmoid(x, alpha=10):
    return 1 / (1 + np.exp(-alpha * x))

def generate_surface(X, Y, plateaus, sectors, alpha=10):
    # Base conical surface
    p = 1.1  # Adjust this for desired steepness
    Z = ((X - x_goal)**2 + (Y - y_goal)**2) * p

    # Add obstacle plateaus
    for (x_min, x_max, y_min, y_max, h) in plateaus:
        Sx = moat_sigmoid(X - x_min, alpha) * moat_sigmoid(x_max - X, alpha)
        Sy = moat_sigmoid(Y - y_min, alpha) * moat_sigmoid(y_max - Y, alpha)
        S = Sx * Sy
        Z = S * h + (1 - S) * Z

    # Add enemy plateaus
    for (x_c, y_c, r_max, theta_min, theta_max, h) in sectors:
        # Convert Cartesian to polar coordinates relative to sector center
        r = np.sqrt((X - x_c)**2 + (Y - y_c)**2)
        theta = np.arctan2(Y - y_c, X - x_c)  # Angle in radians
        
        # Ensure theta is within the range [theta_min, theta_max]
        theta_mask = sigmoid(theta - theta_min, alpha) * sigmoid(theta_max - theta, alpha)
        r_mask = sigmoid(r_max - r, alpha)  # Smooth radius cutoff
        
        S = theta_mask * r_mask  # Combined sector mask
        Z = S * h + (1 - S) * Z  # Blend with terrain

    return Z

# Generate grid
x = np.linspace(0, 20, 100)
y = np.linspace(0, 20, 100)
X, Y = np.meshgrid(x, y)

# Define square plateaus [(x_min, x_max, y_min, y_max, height)]
plateaus = [
    (15, 16, 15, 17, height),
    (4, 5, 3, 5, height)
]
# Define sector plateaus [(x_c, y_c, r_max, theta_min, theta_max, height)]
sectors = [
    (3, 11, 5, -np.pi/6 - np.pi/2, np.pi/6 - np.pi/2, height)
]

# Compute terrain with plateaus and sectors
Z = generate_surface(X, Y, plateaus, sectors)


# Plot 3D Surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Cost (Height)')
ax.set_title('Gradient Descent Terrain with Obstacles & Smooth Enemy FOV Sectors')

plt.show()

# Compute gradient using finite differences
dx = np.gradient(Z, axis=1)
dy = np.gradient(Z, axis=0)

# Gradient descent navigation
start_x, start_y = 0, 0
alpha = 0.2  # Step size
max_steps = 10000

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
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)

# Plot enemy positions and directions
plt.scatter([x_goal], [y_goal], color='red', marker='X', s=100, label="Goal")
# plt.scatter(*zip(*obstacle_centers), color='white', marker='o', s=100, label="Obstacles")
for (ex, ey, _, _, _, _) in sectors:
    plt.scatter(ex, ey, color='red', marker='o', s=100)  # Enemy positions

# Plot agent path
plt.plot(path_x, path_y, color='cyan', linestyle='-', marker='o', markersize=3, label="Agent Path")
plt.scatter([start_x], [start_y], color='blue', marker='o', s=100, label="Start")

plt.legend()
plt.title("Gradient Descent Pathfinding with Smooth Plateau FOV")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()





quit()





# Plot the terrain
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

plt.show()

# Optional: 2D Contour Plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)

# Plot enemy positions and directions
plt.legend()
plt.title("Contour Map of Terrain with Smooth Enemy FOV Sectors")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()