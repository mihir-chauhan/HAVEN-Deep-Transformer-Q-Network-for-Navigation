from ast import Pass
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from scipy import ndimage
from sklearn.cluster import DBSCAN
import threading
import random
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# Conditional ROS imports
try:
    import rospy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    #print("!!!!! ROS not available, running in simulation mode only")

# --------------------------
# Global Environment Configuration
# --------------------------
# Environment bounds in robot-centric coordinates
ENVIRONMENT_MIN = -5
ENVIRONMENT_MAX = 15

class ROSDataLocal:
    def __init__(self, scene_min=None, scene_max=None, use_ros=True, num_enemies=5):
        # Use global environment bounds if not specified
        self.scene_min = scene_min if scene_min is not None else ENVIRONMENT_MIN
        self.scene_max = scene_max if scene_max is not None else ENVIRONMENT_MAX
        self.unitMultiplier = 100.0  # m -> cm
        self.odom = [0.0, 0.0, 0.0]  # [x, y, heading]
        self.pointCloud = []
        self.cmdvel = [0.0, 0.0]  # [linear_vel, angular_vel]
        self.clusters = []
        self.obsBb = []
        self.obstacles = []  # Will store Shapely Polygon objects
        self.resolution = 5
        self.grid_size = 1
        self.width = int((self.scene_max - self.scene_min) * self.unitMultiplier / self.grid_size)
        self.height = int((self.scene_max - self.scene_min) * self.unitMultiplier / self.grid_size)
        self.area = self.width * self.height * ((self.resolution/100.0)**2)
        
        # Robot-centric coordinate system
        self.robot_start_pos = None  # Will be set when we first get odometry
        self.robot_current_pos = np.array([0.0, 0.0])  # Current position in robot frame
        self.robot_start_heading = 0.0  # Initial heading
        
        # Temporal filtering for better obstacle detection
        self.recent_point_clouds = []  # Store recent point clouds for temporal filtering
        self.max_point_cloud_history = 2  # Keep last 2 point clouds (reduced from 3)
        self.max_points_per_cloud = 10000  # Limit points per cloud to prevent memory explosion
        
        # Point cloud tracking
        self.pointcloud_counter = 0  # Track number of point clouds received
        self.last_processed_counter = -1  # Track last processed point cloud
        
        # Simulation mode toggle
        self.use_ros = use_ros
        
        # Enemy tracking
        self.enemy_positions = {}  # Dictionary to store enemy positions
        self.enemy_headings = {}   # Dictionary to store enemy headings
        self.enemy_lock = threading.Lock()
        
        # Initialize ROS node if using ROS
        if self.use_ros and ROS_AVAILABLE:
            if not rospy.core.is_initialized():
                rospy.init_node('stealth_agent', anonymous=True)
                
            # Initialize publishers and subscribers using the new pattern (agent topics)
            self.cmd_vel_pub = self.initializeTwistPublisher('/husky/husky_velocity_controller/cmd_vel')
            self.initializePointCloudPublisher('/husky/lidar_points')
            self.initializeOdometryPublisher('/husky/husky_velocity_controller/odom')
            
            # Enemy state storage (names keyed)
            self.enemy_positions = {}   # name -> [x_robot, y_robot]
            self.enemy_headings = {}    # name -> heading_robot
            self.enemy_spawn_world_pos = {}  # name -> np.array([wx, wy])
            self.enemy_odom_start_pos = {}   # name -> np.array([x0, y0]) in enemy's odom frame
            self.enemy_prev_world_pos = {}   # name -> np.array([wx, wy])
            self.enemy_est_world_heading = {} # name -> float
        else:
            # Initialize simulation state
            self.sim_time = 0.0
            self.sim_dt = 0.1  # Simulation time step
            self.sim_agent_pos = np.array([self.odom[0], self.odom[1]])
            self.sim_agent_heading = self.odom[2]
            self.sim_enemy_positions = {}
            self.sim_enemy_headings = {}
            self.sim_enemy_speeds = {}
            self.sim_enemy_directions = {}
            
            # Initialize enemies in simulation mode
            for i in range(5):
                x = random.uniform(self.scene_min + 2, self.scene_max)
                y = random.uniform(self.scene_min + 2, self.scene_max)
                pos = np.array([x, y])
                heading = random.uniform(0, 2 * math.pi)
                self.sim_enemy_positions[i] = pos
                self.sim_enemy_headings[i] = heading
                self.sim_enemy_speeds[i] = 0.3
                self.sim_enemy_directions[i] = np.array([math.cos(heading), math.sin(heading)])
            
            # Generate random obstacles for simulation mode
            self.generate_random_obstacles()

    def initializeTwistPublisher(self, topicName):
        if ROS_AVAILABLE:
            return rospy.Publisher(topicName, Twist, queue_size=1)
        else:
            return None

    def initializePointCloudPublisher(self, topicName):
        if ROS_AVAILABLE:
            rospy.Subscriber(topicName, PointCloud2, self.pointcloud_callback)

    def initializeOdometryPublisher(self, topicName):
        if ROS_AVAILABLE:
            rospy.Subscriber(topicName, Odometry, self.odom_callback)

    def generate_random_obstacles(self, num_obstacles=6):
        """Generate random obstacles for simulation mode"""
        self.obstacles = []
        for _ in range(num_obstacles):
            # Generate random polygon vertices
            num_vertices = random.randint(3, 6)
            center_x = random.uniform(self.scene_min + 1, self.scene_max - 1)
            center_y = random.uniform(self.scene_min + 1, self.scene_max - 1)
            radius = random.uniform(0.5, 2.0)
            
            # Generate vertices in a circle around the center
            angles = np.sort([random.uniform(0, 2*math.pi) for _ in range(num_vertices)])
            vertices = []
            for angle in angles:
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                vertices.append((x, y))
            
            # Create polygon and add to obstacles
            poly = Polygon(vertices)
            self.obstacles.append(poly)

    def generate_sample_pointcloud(self):
        """Generate sample point cloud data for testing in simulation mode (in meters)"""
        #print("!!!!! Generating sample point cloud data...")
        sample_points = []
        
        # Generate points around existing obstacles
        for obs in self.obstacles:
            coords = np.array(obs.exterior.coords)
            for point in coords:
                # Add some noise around the obstacle boundary
                for _ in range(20):  # 20 points per obstacle vertex
                    noise_x = random.uniform(-0.2, 0.2)
                    noise_y = random.uniform(-0.2, 0.2)
                    noise_z = random.uniform(0, 2.0)
                    sample_points.append([
                        (point[0] + noise_x),  # meters
                        (point[1] + noise_y),  # meters
                        noise_z
                    ])
        
        # Add some random noise points
        for _ in range(50):
            x = random.uniform(self.scene_min, self.scene_max)
            y = random.uniform(self.scene_min, self.scene_max)
            z = random.uniform(0, 3.0)
            sample_points.append([x, y, z])
        
        #print(f"!!!!! Generated {len(sample_points)} sample point cloud points")
        return sample_points

    def get_obstacles(self):
        """Get current obstacles as Shapely Polygon objects"""
        return self.obstacles

    def update_simulation(self):
        """Update simulation state when not using ROS"""
        if not self.use_ros:
            # Update agent position based on velocity commands
            linear_vel, angular_vel = self.cmdvel
            self.sim_agent_heading += angular_vel * self.sim_dt
            direction = np.array([math.cos(self.sim_agent_heading), math.sin(self.sim_agent_heading)])
            self.sim_agent_pos += direction * linear_vel * self.sim_dt
            
            # Update odometry
            self.odom = [self.sim_agent_pos[0], self.sim_agent_pos[1], self.sim_agent_heading]
            
            # Update enemy positions
            for i in range(5):
                # Simple random movement for enemies
                heading = self.sim_enemy_headings[i]
                direction = np.array([math.cos(heading), math.sin(heading)])
                self.sim_enemy_positions[i] += direction * self.sim_enemy_speeds[i] * self.sim_dt
                
                # Random heading changes
                if random.random() < 0.1:  # 10% chance to change direction
                    self.sim_enemy_headings[i] += random.uniform(-0.1, 0.1)
                
                # Update enemy direction
                self.sim_enemy_directions[i] = np.array([
                    math.cos(self.sim_enemy_headings[i]),
                    math.sin(self.sim_enemy_headings[i])
                ])
            
            self.sim_time += self.sim_dt

    def enemy_pose_callback(self, msg, enemy_id):
        """Callback for enemy pose updates"""
        with self.enemy_lock:
            self.enemy_positions[enemy_id] = [
                msg.pose.position.x,
                msg.pose.position.y
            ]
            # Convert quaternion to heading
            q = msg.pose.orientation
            heading = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.enemy_headings[enemy_id] = heading

    def pointcloud_callback(self, msg):
        """Callback for point cloud updates"""
        # Convert PointCloud2 to list of points
        points = []
        
        # Import point_cloud2 for proper parsing
        import sensor_msgs.point_cloud2 as pc2
        
        # Parse PointCloud2 message
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point
            points.append([x, y, z])
        
        self.pointcloud_counter += 1
        print(f"Received {len(points)} points from point cloud callback (#{self.pointcloud_counter})")
        self.pointCloud = points
        # Don't process here - let the main loop handle processing for better control

    def odom_callback(self, msg):
        """Callback for odometry updates"""
        # Get current odometry in world frame
        world_x = msg.pose.pose.position.x
        world_y = msg.pose.pose.position.y
        world_heading = math.atan2(
            2.0 * (msg.pose.pose.orientation.w * msg.pose.pose.orientation.z),
            1.0 - 2.0 * (msg.pose.pose.orientation.z * msg.pose.pose.orientation.z)
        )
        # Set robot start position on first odometry reading
        if self.robot_start_pos is None:
            self.robot_start_pos = np.array([world_x, world_y])
            self.robot_start_heading = world_heading
            print(f"Robot start position set to: [{world_x:.2f}, {world_y:.2f}]")
        
        # Calculate robot position in robot-centric frame
        robot_world_pos = np.array([world_x, world_y])
        robot_relative_pos = robot_world_pos - self.robot_start_pos

        
        # Rotate to robot-centric frame
        cos_heading = math.cos(-self.robot_start_heading)
        sin_heading = math.sin(-self.robot_start_heading)
        rotation_matrix = np.array([[cos_heading, -sin_heading], [sin_heading, cos_heading]])
        self.robot_current_pos = np.dot(rotation_matrix, robot_relative_pos)
        
        # Update odometry for compatibility
        self.odom = [self.robot_current_pos[0], self.robot_current_pos[1], world_heading - self.robot_start_heading]

    def world_to_robot_frame(self, world_point):
        """Convert a point from world frame to robot-centric frame"""
        if self.robot_start_pos is None:
            return world_point
        
        # Translate to robot-relative coordinates
        relative_pos = np.array(world_point) - self.robot_start_pos
        
        # Rotate to robot-centric frame
        cos_heading = math.cos(-self.robot_start_heading)
        sin_heading = math.sin(-self.robot_start_heading)
        rotation_matrix = np.array([[cos_heading, -sin_heading], [sin_heading, cos_heading]])
        return np.dot(rotation_matrix, relative_pos)

    def robot_to_world_frame(self, robot_point):
        """Convert a point from robot-centric frame to world frame"""
        if self.robot_start_pos is None:
            return robot_point
        
        # Rotate back to world frame
        cos_heading = math.cos(self.robot_start_heading)
        sin_heading = math.sin(self.robot_start_heading)
        rotation_matrix = np.array([[cos_heading, -sin_heading], [sin_heading, cos_heading]])
        world_relative = np.dot(rotation_matrix, robot_point)
        
        # Translate back to world coordinates
        return world_relative + self.robot_start_pos

    def get_enemy_data(self):
        """Get current enemy positions and headings"""
        if self.use_ros:
            with self.enemy_lock:
                return self.enemy_positions.copy(), self.enemy_headings.copy()
        else:
            return self.sim_enemy_positions.copy(), self.sim_enemy_headings.copy()

    def publish_velocity(self, linear_vel, angular_vel):
        """Publish velocity commands to ROS or update simulation"""
        self.cmdvel = [linear_vel, angular_vel]
        if self.use_ros and ROS_AVAILABLE and self.cmd_vel_pub is not None:
            twist = Twist()
            twist.linear.x = linear_vel
            twist.angular.z = angular_vel
            self.cmd_vel_pub.publish(twist)
        else:
            self.update_simulation()

    def get_cmdvel(self):
        return self.cmdvel
    
    def get_pointcloud(self):
        return self.pointCloud
    
    def has_new_pointcloud(self):
        """Check if there's new point cloud data to process"""
        return self.pointcloud_counter > self.last_processed_counter
    
    def mark_pointcloud_processed(self):
        """Mark current point cloud as processed"""
        self.last_processed_counter = self.pointcloud_counter
    
    def cleanup_memory(self):
        """Clean up memory by clearing old data"""
        # Clear old point cloud history if getting too large
        if len(self.recent_point_clouds) > self.max_point_cloud_history:
            self.recent_point_clouds = self.recent_point_clouds[-self.max_point_cloud_history:]
        
        # Clear old clusters
        if hasattr(self, 'clusters') and len(self.clusters) > 100:
            self.clusters = []
        
        # Force garbage collection periodically
        import gc
        gc.collect()

    def get_odom(self):
        return self.odom

    def set_cmdvel(self, linear_vel, angular_vel):
        self.cmdvel = [linear_vel, angular_vel]

    def set_odom(self, x, y, heading):
        self.odom = [x, y, heading]

    def dbscan_clustering(self, twoDCoords):
        data = np.array(twoDCoords)
        if data.size == 0:
            #print("No points to cluster.")
            return []
        
        # Improved DBSCAN parameters for better large obstacle detection
        # Larger eps to handle sparser point clouds from moving robot
        # Lower min_samples to detect obstacles with fewer points
        # Optimized parameters for better performance
        db = DBSCAN(eps=1.0, min_samples=4, n_jobs=1).fit(data)  # Use single thread to avoid memory issues
        labels = db.labels_
        unique_labels = np.unique(labels)
        clusters = []
        
        # Also add single large obstacles (noise points) as individual clusters
        noise_points = data[labels == -1]
        if len(noise_points) > 0:
            # Group nearby noise points into clusters
            if len(noise_points) >= 2:
                noise_db = DBSCAN(eps=2.0, min_samples=2).fit(noise_points)
                noise_labels = noise_db.labels_
                for noise_label in np.unique(noise_labels):
                    if noise_label != -1:  # Skip noise within noise
                        noise_cluster = noise_points[noise_labels == noise_label]
                        if len(noise_cluster) >= 2:
                            clusters.append(noise_cluster.T)
        
        # Add regular clusters
        for label in unique_labels:
            if label != -1:  # Skip noise points (already handled above)
                cluster_points = data[labels == label]
                clusters.append(cluster_points.T)
        
        return clusters

    def cluster_bounding_box(self, dbscanOutput):
        x_arr = dbscanOutput[0]
        y_arr = dbscanOutput[1]
        return [min(x_arr), min(y_arr), max(x_arr), max(y_arr)]
    
    def convert_xy_to_field_centric(self, pointList):
        shiftedPointList = []
        robotPose = self.get_odom()
        for point in pointList:
            newPointX = point[0] + robotPose[0]
            newPointY = point[1] + robotPose[1]
            shiftedPointList.append((newPointX, newPointY))
        return shiftedPointList

    def process_pointcloud(self, points):
        """
        Process a list of [x, y, z] points into obstacles in robot-centric coordinates
        Points are expected to be in meters
        """
        # --- Occupancy-grid pipeline ---
        # Temporal fusion (bounded) and downsampling
        if len(points) > self.max_points_per_cloud:
            indices = np.random.choice(len(points), self.max_points_per_cloud, replace=False)
            points = [points[i] for i in indices]
            print(f"Downsampled point cloud to {self.max_points_per_cloud} points")
        self.recent_point_clouds.append(points)
        if len(self.recent_point_clouds) > self.max_point_cloud_history:
            self.recent_point_clouds.pop(0)
        all_recent_points = []
        for pc in self.recent_point_clouds:
            all_recent_points.extend(pc)
        if len(all_recent_points) > self.max_points_per_cloud * 2:
            indices = np.random.choice(len(all_recent_points), self.max_points_per_cloud * 2, replace=False)
            all_recent_points = [all_recent_points[i] for i in indices]
        print(f"Processing {len(all_recent_points)} fused points from {len(self.recent_point_clouds)} frames...")

        pts = np.asarray(all_recent_points, dtype=float)
        if pts.size == 0:
            self.obstacles = []
            return
        # Transform to robot-centric frame by shifting with current robot pos
        rx, ry = self.robot_current_pos[0], self.robot_current_pos[1]
        x = pts[:, 0] + rx
        y = pts[:, 1] + ry
        z = pts[:, 2]
        # Bounds and z filter
        min_z, max_z = -0.2, 2.0
        within_bounds = (
            (x > self.scene_min) & (x < self.scene_max) &
            (y > self.scene_min) & (y < self.scene_max) &
            (z > min_z) & (z < max_z)
        )
        x, y = x[within_bounds], y[within_bounds]
        if x.size == 0:
            self.obstacles = []
            return
        xy = np.column_stack([x, y])

        # Radius outlier removal
        try:
            tree = cKDTree(xy)
            counts = tree.query_ball_point(xy, r=0.25, return_length=True)
            mask = counts >= 3
            xy = xy[mask]
        except Exception:
            pass
        if xy.shape[0] == 0:
            self.obstacles = []
            return

        # Occupancy grid build
        res = 0.1
        extent = self.scene_max - self.scene_min
        W = int(math.ceil(extent / res))
        H = int(math.ceil(extent / res))
        grid = np.zeros((H, W), dtype=bool)
        ix = np.floor((xy[:, 0] - self.scene_min) / res).astype(int)
        iy = np.floor((xy[:, 1] - self.scene_min) / res).astype(int)
        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        grid[iy[valid], ix[valid]] = True

        # Morphological cleanup
        struct = ndimage.generate_binary_structure(2, 1)
        grid = ndimage.binary_closing(grid, structure=struct, iterations=3)
        grid = ndimage.binary_opening(grid, structure=struct, iterations=2)

        # Inflation
        inflate_cells = max(1, int(round(0.4 / res)))
        yy, xx = np.ogrid[-inflate_cells:inflate_cells+1, -inflate_cells:inflate_cells+1]
        circular = (xx*xx + yy*yy) <= (inflate_cells*inflate_cells)
        grid = ndimage.binary_dilation(grid, structure=circular)

        # Connected components and polygonization
        labeled, num = ndimage.label(grid)
        obstacles = []
        min_area_m2 = 0.05
        max_area_m2 = 100.0
        for label_id in range(1, num + 1):
            mask_comp = (labeled == label_id)
            if not mask_comp.any():
                continue
            ys, xs = np.where(mask_comp)
            cell_boxes = []
            for r, c in zip(ys, xs):
                minx = self.scene_min + c * res
                miny = self.scene_min + r * res
                maxx = minx + res
                maxy = miny + res
                cell_boxes.append(box(minx, miny, maxx, maxy))
            if not cell_boxes:
                continue
            merged = unary_union(cell_boxes)
            geoms = [merged] if merged.geom_type == 'Polygon' else list(merged.geoms)
            for geom in geoms:
                area = geom.area
                if area < min_area_m2 or area > max_area_m2:
                    continue
                poly = geom.simplify(res * 0.5, preserve_topology=True)
                if poly.is_valid and not poly.is_empty:
                    obstacles.append(poly)
        self.obstacles = obstacles
        print(f"Occupancy-grid obstacles: {len(self.obstacles)} polygons")
    
    def detect_point_cloud_edges(self, points):
        """
        Detect edges in point cloud that might indicate obstacles
        Returns list of synthetic obstacle points representing detected edges
        """
        if len(points) < 10:
            return []
            
        edge_obstacles = []
        points_array = np.array(points)
        
        # Limit processing to prevent memory explosion
        if len(points_array) > 5000:  # Sample down if too many points
            indices = np.random.choice(len(points_array), 5000, replace=False)
            points_array = points_array[indices]
        
        # Sort points by angle from robot center
        angles = np.arctan2(points_array[:, 1], points_array[:, 0])
        sorted_indices = np.argsort(angles)
        sorted_points = points_array[sorted_indices]
        sorted_angles = angles[sorted_indices]
        
        # Look for large gaps in radial distance that might indicate obstacles
        # Process every 10th point to reduce computation and memory usage
        step_size = max(1, len(sorted_points) // 100)  # Process at most 100 edge checks
        
        for i in range(0, len(sorted_points) - step_size, step_size):
            current_point = sorted_points[i]
            next_point = sorted_points[min(i + step_size, len(sorted_points) - 1)]
            
            current_dist = np.linalg.norm(current_point[:2])
            next_dist = np.linalg.norm(next_point[:2])
            angle_diff = abs(sorted_angles[min(i + step_size, len(sorted_points) - 1)] - sorted_angles[i])
            
            # If there's a sudden jump in distance and reasonable angle difference,
            # it might indicate an obstacle creating a "shadow" in the point cloud
            if 0.1 < angle_diff < 0.5 and abs(current_dist - next_dist) > 4.0:
                # Create synthetic obstacle points to fill the gap
                min_dist = min(current_dist, next_dist)
                if 2.0 < min_dist < 15.0:  # Only for obstacles in reasonable range
                    # Create a small obstacle at the closer point (just 1 point, not 9)
                    closer_point = current_point if current_dist < next_dist else next_point
                    edge_obstacles.append([closer_point[0], closer_point[1], 1.0])
                    
                    # Limit total synthetic points to prevent memory explosion
                    if len(edge_obstacles) > 50:  # Maximum 50 synthetic points
                        break
        
        return edge_obstacles
    
    def cluster_filtered_pointcloud(self, filteredPointCloud, original_points=None, edge_obstacles=None):
        #print("Starting DBSCAN clustering...")
        
        # Combine regular points with edge-detected obstacle points
        all_points = filteredPointCloud.copy()
        if edge_obstacles:
            all_points.extend(edge_obstacles)
            #print(f"Added {len(edge_obstacles)} edge-detected obstacle points")
        
        # Limit total points for clustering to prevent memory issues
        if len(all_points) > 15000:  # Maximum points for clustering
            indices = np.random.choice(len(all_points), 15000, replace=False)
            all_points = [all_points[i] for i in indices]
            print(f"Limited clustering input to 15000 points from {len(filteredPointCloud) + len(edge_obstacles if edge_obstacles else [])}")
        
        self.clusters = self.dbscan_clustering(all_points)
        #print(f"Found {len(self.clusters)} clusters")
        
        self.obstacles = []  # Reset obstacles list
        self.obsBb = []
        valid_clusters = 0
        
        for i, cluster in enumerate(self.clusters):
            #print(f"Processing cluster {i+1} with {len(cluster[0])} points...")
            clusterBoundingBox = self.cluster_bounding_box(cluster)
            self.obsBb.append(clusterBoundingBox)
            
            # Remove clusters that are too large by area (but allow larger obstacles like boats)
            cluster_area = abs(clusterBoundingBox[2] - clusterBoundingBox[0]) * abs(clusterBoundingBox[3] - clusterBoundingBox[1])
            max_area = 200  # Increased from 50 to allow larger obstacles like boats
            min_area = 0.5  # Minimum area to avoid tiny noise clusters
            
            #print(f"  Cluster {i+1} area: {cluster_area:.3f}, allowed range: {min_area:.3f} - {max_area:.3f}")
            
            if min_area < cluster_area < max_area:
                # If obstacle is too small, ignore
                if (len(cluster[0])) > 3:
                    #print(f"  Cluster {i+1} is valid, creating convex hull...")
                    cluster2dTransposed = np.array([cluster[0], cluster[1]]).T
                    hull = ConvexHull(cluster2dTransposed)
                    
                    # Create Shapely polygon from convex hull
                    hull_vertices = cluster2dTransposed[hull.vertices]
                    poly = Polygon(hull_vertices)
                    
                    # Add to obstacles list
                    self.obstacles.append(poly)
                    #print(f"  Cluster {i+1} converted to Shapely polygon with {len(hull_vertices)} vertices")
                    valid_clusters += 1
                else:
                    #print(f"  Cluster {i+1} too small ({len(cluster[0])} points), skipping")
                    pass
            else:
                #print(f"  Cluster {i+1} too large, skipping")
                pass
        
        #print(f"Processing complete: {valid_clusters} valid obstacles created")
        pass

    def initialize_enemy_publishers_and_subscribers(self, enemy_names):
        """Initialize cmd_vel publishers and odom subscribers for each enemy husky"""
        if not (self.use_ros and ROS_AVAILABLE):
            return {}, {}
        import rospy
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        enemy_cmd_vel_pubs = {}
        enemy_odom_subs = {}
        # Initialize storage
        for name in enemy_names:
            self.enemy_positions[name] = [0.0, 0.0]
            self.enemy_headings[name] = 0.0
        # Odom callback using odom-delta + spawn world pos, heading from velocity
        def make_odom_callback(name):
            def cb(msg):
                odom_x = msg.pose.pose.position.x
                odom_y = msg.pose.pose.position.y
                if name not in self.enemy_odom_start_pos:
                    self.enemy_odom_start_pos[name] = np.array([odom_x, odom_y])
                odom_delta = np.array([odom_x, odom_y]) - self.enemy_odom_start_pos[name]
                if name in self.enemy_spawn_world_pos:
                    p_world = self.enemy_spawn_world_pos[name] + odom_delta
                else:
                    p_world = np.array([odom_x, odom_y])
                if name in self.enemy_prev_world_pos:
                    v_world = p_world - self.enemy_prev_world_pos[name]
                    v_norm = np.linalg.norm(v_world)
                    if v_norm > 1e-3:
                        self.enemy_est_world_heading[name] = math.atan2(v_world[1], v_world[0])
                if name not in self.enemy_est_world_heading:
                    q = msg.pose.pose.orientation
                    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                    self.enemy_est_world_heading[name] = yaw
                self.enemy_prev_world_pos[name] = p_world.copy()
                # Convert world -> robot-centric
                if self.robot_start_pos is not None:
                    pos_robot = self.world_to_robot_frame(p_world)
                    heading_robot = self.enemy_est_world_heading[name] - self.robot_start_heading
                    while heading_robot > math.pi:
                        heading_robot -= 2 * math.pi
                    while heading_robot < -math.pi:
                        heading_robot += 2 * math.pi
                else:
                    pos_robot = p_world
                    heading_robot = self.enemy_est_world_heading[name]
                self.enemy_positions[name] = [pos_robot[0], pos_robot[1]]
                self.enemy_headings[name] = heading_robot
            return cb
        for name in enemy_names:
            cmd_topic = f"/{name}/husky_velocity_controller/cmd_vel"
            odom_topic = f"/{name}/husky_velocity_controller/odom"
            enemy_cmd_vel_pubs[name] = rospy.Publisher(cmd_topic, Twist, queue_size=1)
            enemy_odom_subs[name] = rospy.Subscriber(odom_topic, Odometry, make_odom_callback(name))
        return enemy_cmd_vel_pubs, enemy_odom_subs

    def set_enemy_spawn_world_positions(self, name_to_world_xy):
        """Provide spawn world positions to enable odom-delta localization (name -> [wx, wy])."""
        for name, xy in name_to_world_xy.items():
            self.enemy_spawn_world_pos[name] = np.array(xy, dtype=float)

    def get_enemy_states(self):
        """Return dict name -> (pos_robot_frame np.array([x,y]), heading_robot_frame float)."""
        states = {}
        for name, pos in self.enemy_positions.items():
            states[name] = (np.array(pos, dtype=float), float(self.enemy_headings.get(name, 0.0)))
        return states

    def publish_enemy_velocity(self, name, linear_vel, angular_vel, pubs):
        if self.use_ros and ROS_AVAILABLE and name in pubs:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.x = float(linear_vel)
            twist.angular.z = float(angular_vel)
            pubs[name].publish(twist)

    def get_enemy_position(self, name):
        return self.enemy_positions.get(name, [0.0, 0.0])

    def get_enemy_heading(self, name):
        return self.enemy_headings.get(name, 0.0)

    def publish_enemy_velocity(self, name, linear_vel, angular_vel, enemy_cmd_vel_pubs):
        if self.use_ros and ROS_AVAILABLE and name in enemy_cmd_vel_pubs:
            from geometry_msgs.msg import Twist
            twist = Twist()
            twist.linear.x = linear_vel
            twist.angular.z = angular_vel
            enemy_cmd_vel_pubs[name].publish(twist)