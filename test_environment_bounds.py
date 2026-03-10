#!/usr/bin/env python3
"""
Simple script to get robot position and create 3D plot of raw point cloud data
"""

import rospy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def main():
    """Main function to get robot position and plot point cloud"""
    print("Starting Robot Position and Point Cloud 3D Plot")
    print("=" * 50)
    
    try:
        # Import ROS data handler
        from ros_data_local import ROSDataLocal
        
        # Initialize ROS node
        if not rospy.core.is_initialized():
            rospy.init_node('test_3d_pointcloud', anonymous=True)
        
        # Create ROS data handler
        ros_data = ROSDataLocal(use_ros=True)
        
        # Wait for ROS to be ready
        print("Waiting for ROS to be ready...")
        time.sleep(2)
        
        # Check if ROS topics are available
        print("\nChecking ROS topics...")
        try:
            topics = rospy.get_published_topics()
            print(f"Available topics: {len(topics)}")
            for topic_name, topic_type in topics:
                if 'odom' in topic_name or 'husky' in topic_name:
                    print(f"  - {topic_name}: {topic_type}")
        except Exception as e:
            print(f"Could not get ROS topics: {e}")
        
        # Get robot position
        print("\nGetting Robot Position:")
        robot_pos = ros_data.get_odom()
        print(f"Robot position: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}]")
        print(f"Robot heading: {robot_pos[2]:.2f} radians")
        
        # Check if robot_start_pos is set (indicates if odometry callback was called)
        if hasattr(ros_data, 'robot_start_pos'):
            if ros_data.robot_start_pos is None:
                print("⚠️  Warning: robot_start_pos is None - no odometry data received yet")
                print("   This means the robot position is using default values")
            else:
                print(f"✓ Robot start position set: [{ros_data.robot_start_pos[0]:.2f}, {ros_data.robot_start_pos[1]:.2f}]")
        
        # Wait a bit more for potential odometry data
        print("\nWaiting for potential odometry data...")
        time.sleep(3)
        
        # Get updated robot position
        robot_pos_updated = ros_data.get_odom()
        print(f"Updated robot position: [{robot_pos_updated[0]:.2f}, {robot_pos_updated[1]:.2f}]")
        print(f"Updated robot heading: {robot_pos_updated[2]:.2f} radians")
        
        # Get raw point cloud data
        print("\nGetting Raw Point Cloud Data:")
        point_cloud = ros_data.get_pointcloud()
        while not point_cloud:
            print("Waiting for point cloud data...")
            time.sleep(1)
            point_cloud = ros_data.get_pointcloud()
            print(f"Point cloud: {point_cloud}")
        
        if point_cloud:
            print(f"Found {len(point_cloud)} point cloud points")
            
            # Convert to numpy array for easier plotting
            points_array = np.array(point_cloud)
            
            # Create 3D plot
            print("Creating 3D plot...")
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot point cloud points
            if len(points_array) > 0:
                ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                          c='blue', s=1, alpha=0.6, label='Point Cloud')
            
            # Plot robot position
            ax.scatter(robot_pos[0], robot_pos[1], 0, 
                      c='red', s=100, marker='o', label='Robot')
            
            # Set labels and title
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title('3D Point Cloud Visualization')
            
            # Set axis limits based on environment bounds
            ax.set_xlim(ros_data.scene_min, ros_data.scene_max)
            ax.set_ylim(ros_data.scene_min, ros_data.scene_max)
            ax.set_zlim(0, 10)  # Assuming max height of 10m
            
            ax.legend()
            
            # Show the plot
            plt.tight_layout()
            plt.show()
            print("✓ 3D plot displayed successfully")
            
        else:
            print("✗ No point cloud data available")
            
            # If no point cloud data, generate sample data for visualization
            if hasattr(ros_data, 'generate_sample_pointcloud'):
                print("Generating sample point cloud data for visualization...")
                sample_points = ros_data.generate_sample_pointcloud()
                if sample_points:
                    points_array = np.array(sample_points)
                    
                    # Create 3D plot with sample data
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    ax.scatter(points_array[:, 0], points_array[:, 1], points_array[:, 2], 
                              c='blue', s=1, alpha=0.6, label='Sample Point Cloud')
                    ax.scatter(robot_pos[0], robot_pos[1], 0, 
                              c='red', s=100, marker='o', label='Robot')
                    
                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Y (m)')
                    ax.set_zlabel('Z (m)')
                    ax.set_title('3D Point Cloud Visualization (Sample Data)')
                    ax.set_xlim(ros_data.scene_min, ros_data.scene_max)
                    ax.set_ylim(ros_data.scene_min, ros_data.scene_max)
                    ax.set_zlim(0, 10)
                    ax.legend()
                    
                    plt.tight_layout()
                    plt.show()
                    print("✓ 3D plot with sample data displayed successfully")
            
            return False
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    main() 