import os
import open3d as o3d
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation as R

DATASET_CSV_PATH = 'output/2025_07_24_11_00_00/dataset.csv'
DATASET_CSV_COLUMNS = [
    'frame_index',
    'original_path',
    'rgb_frame_path',
    'depth_frame_path',
    'depth_confidence_frame_path',
    # intrinsics matrix
    'intrinsics_00', 'intrinsics_01', 'intrinsics_02',
    'intrinsics_10', 'intrinsics_11', 'intrinsics_12',
    'intrinsics_20', 'intrinsics_21', 'intrinsics_22',
    # odometry data
    'odometry_timestamp',
    'odometry_x', 'odometry_y', 'odometry_z',
    'odometry_qx', 'odometry_qy', 'odometry_qz', 'odometry_qw',
    # imu data
    'imu_timestamp',
    'a_x', 'a_y', 'a_z',
    'alpha_x', 'alpha_y', 'alpha_z',
    # location data
    'location_timestamp',
    'latitude', 'longitude', 'altitude',
    'horizontal_accuracy', 'vertical_accuracy', 
    'speed', 'course', 'floor_level',
    # heading data
    'heading_timestamp',
    'magnetic_heading', 'true_heading', 'heading_accuracy'
]

dataset_df = pd.read_csv(DATASET_CSV_PATH, header=0, names=DATASET_CSV_COLUMNS)
# print(f"Loaded dataset with {len(dataset_df)} entries.")

def get_intrinsics(data: pd.Series) -> np.ndarray:
    """Extracts the camera intrinsics matrix from a DataFrame row."""
    intrinsics = np.array([
        [data['intrinsics_00'], data['intrinsics_01'], data['intrinsics_02']],
        [data['intrinsics_10'], data['intrinsics_11'], data['intrinsics_12']],
        [data['intrinsics_20'], data['intrinsics_21'], data['intrinsics_22']]
    ])
    return intrinsics

# intrinsics = get_intrinsics(dataset_df.iloc[0])
# print("Camera intrinsics matrix:")
# print(intrinsics)

def get_pose_matrix(data: pd.Series) -> np.ndarray:
    """Constructs the pose matrix from odometry data."""
    translation = np.array([data['odometry_x'], data['odometry_y'], data['odometry_z']])
    rotation = R.from_quat([
        data['odometry_qx'], data['odometry_qy'], data['odometry_qz'], data['odometry_qw']
    ]).as_matrix()
    
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = translation
    return pose_matrix

# pose = get_pose_matrix(dataset_df.iloc[0])
# print("Pose matrix:")
# print(pose)

def depth_to_pointcloud(depth_map, image, K, pose=None):
    """Converts depth map to point cloud in world coordinates."""
    H, W = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map.astype(np.float32) / 1000.0  # Convert mm to meters
    valid = z > 0

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1)[valid]
    colors = image[v[valid], u[valid]].astype(np.float32) / 255.0

    return points, colors

# Visualize the point cloud with zoom
def visualize_pointcloud(points, colors):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.ViewControl.set_front(o3d.visualization.ViewControl, [0, 0, -1])
    o3d.visualization.ViewControl.set_lookat(o3d.visualization.ViewControl, [0, 0, 0])
    o3d.visualization.ViewControl.set_up(o3d.visualization.ViewControl, [0, 1, 0])
    o3d.visualization.ViewControl.set_field_of_view(o3d.visualization.ViewControl, 60.0)

# Example usage
if __name__ == "__main__":
    data_series = dataset_df.iloc[0]
    intrinsics = get_intrinsics(data_series)
    pose = get_pose_matrix(data_series)
    
    # Load an RGB image
    rgb_file = os.path.join(os.path.dirname(DATASET_CSV_PATH), data_series['rgb_frame_path'])
    image = cv2.imread(rgb_file)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {rgb_file}")
    
    # Load a depth map
    depth_file = os.path.join(os.path.dirname(DATASET_CSV_PATH), data_series['depth_frame_path'])
    depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Could not load depth map from {depth_file}")
    depth_map = cv2.resize(depth_map, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    depth_map = depth_map.astype(np.float32) / 1000.0  # Ensure depth map is float for processing

    # Convert depth map to point cloud
    points, colors = depth_to_pointcloud(depth_map, image, intrinsics, pose)
    visualize_pointcloud(points, colors)