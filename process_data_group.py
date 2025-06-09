import os
import open3d as o3d
import numpy as np
np.float = np.float64
np.int = np.int_
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
# import skvideo.io
import cv2
import platform

description = """
This script processes a group of datasets collected using the Stray Scanner app to create a dataset of 'test' images.
"""

usage = """
Basic usage: python process_group.py <path-to-dataset-folder>
"""

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str, help="Path to StrayScanner dataset to process.")
    parser.add_argument("--every", type=int, default=6, help="Use every nth frame")
    parser.add_argument("--output", type=str, default="output", help="Output directory for processed frames")
    return parser.parse_args()

#################################################################
# Functions to read group of datasets
#################################################################
def get_sub_directories(flags):
    subdirs = [d for d in os.listdir(flags.path) if os.path.isdir(os.path.join(flags.path, d))]
    subdirs = [d for d in subdirs if os.path.exists(os.path.join(flags.path, d, 'rgb.mp4'))]
    # Sort by creation time
    subdirs = sorted(subdirs, key=lambda d: get_creation_time(os.path.join(flags.path, d)))
    print(f"Found {len(subdirs)} subdirectories: {subdirs}")
    return subdirs

def get_creation_time(path):
    """
    Get the creation time of a file or directory.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path)
    else:
        stat = os.stat(path)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime

#################################################################
# Functions to read data from a single dataset
#################################################################
def read_rgb(flags):
    video_path = os.path.join(flags.path, 'rgb.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"RGB video file not found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Total frames in video: {frame_count}, FPS: {fps}")
    rgb_data = {
        'cap': cap,
        'frame_count': frame_count,
        'fps': fps,
    }
    return rgb_data

def read_csv_data(flags):
    # Only 1 intrinsics matrix is expected per dataset.
    intrinsics = np.loadtxt(os.path.join(flags.path, 'camera_matrix.csv'), delimiter=',')
    # The number of rows in the odometry file is expected to be equal to the number of frames in the RGB video.
    odometry = np.loadtxt(os.path.join(flags.path, 'odometry.csv'), delimiter=',', skiprows=1)
    # The number of rows in the imu and location files is expected to be different.
    # Thus, we need to use timestamps from odometry to synchronize them.
    imu = np.loadtxt(os.path.join(flags.path, 'imu.csv'), delimiter=',', skiprows=1)
    location = np.loadtxt(os.path.join(flags.path, 'location.csv'), delimiter=',', skiprows=1)
    heading = np.loadtxt(os.path.join(flags.path, 'heading.csv'), delimiter=',', skiprows=1)

    # The number of depth frames is expected to be equal to the number of frames in the RGB video.
    depth_dir = os.path.join(flags.path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]

    # It is assumed that even if the number of depth and depth confidence frames is different,
    # the files are named in such a way that they correspond to the same sequence number.
    depth_confidence_dir = os.path.join(flags.path, 'confidence')
    depth_confidence_frames = [os.path.join(depth_confidence_dir, p) for p in sorted(os.listdir(depth_confidence_dir))]
    depth_confidence_frames = [f for f in depth_confidence_frames if '.png' in f]

    return {
        'intrinsics': intrinsics, 
        'odometry': odometry,
        'imu': imu,
        'location': location,
        'heading': heading,
        'depth_frames': depth_frames,
        'depth_confidence_frames': depth_confidence_frames
    }

def get_data_sync_params(flags, csv_data):
    """
    The odometry and location data have different timestamp resolutions.
    This function finds the difference in timestamps between the odometry and location data,
    and returns the parameters to synchronize them.

    The scale factor is the same for both the odometry and location data.
    Hence, only time difference is needed.
    """
    odometry_timestamps = csv_data['odometry'][:, 0]
    location_timestamps = csv_data['location'][:, 0]

    # Find the minimum and maximum timestamps
    min_odometry_time = odometry_timestamps.min()
    max_odometry_time = odometry_timestamps.max()
    min_location_time = location_timestamps.min()
    max_location_time = location_timestamps.max()

    # Calculate the time difference
    time_diff1 = min_location_time - min_odometry_time
    time_diff2 = max_location_time - max_odometry_time
    time_diff = (time_diff1 + time_diff2) / 2.0

    return time_diff

def load_depth_file(path, confidence=None, filter_level=0):
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        depth_mm = np.array(Image.open(path))
    # depth_m = depth_mm.astype(np.float32) / 1000.0
    # if confidence is not None:
    #     depth_m[confidence < filter_level] = 0.0
    # return o3d.geometry.Image(depth_m)
    return depth_mm

def load_confidence(path):
    return np.array(Image.open(path))

#################################################################
# Functions to extract frames from the dataset
#################################################################
def create_output_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'rgb')):
        os.makedirs(os.path.join(output_dir, 'rgb'))
    if not os.path.exists(os.path.join(output_dir, 'depth')):
        os.makedirs(os.path.join(output_dir, 'depth'))
    if not os.path.exists(os.path.join(output_dir, 'confidence')):
        os.makedirs(os.path.join(output_dir, 'confidence'))

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

def create_dataset_csv(output_dir, flags):
    csv_path = os.path.join(output_dir, 'dataset.csv')
    if os.path.exists(csv_path):
        print(f"Dataset CSV file already exists at {csv_path}. Overwriting...")
    else:
        with open(csv_path, 'w') as f:
            f.write(','.join(DATASET_CSV_COLUMNS) + '\n')
        print(f"Created dataset CSV file at {csv_path}")
    return csv_path

def extract_frames(flags, rgb_data, csv_data, dataset_csv_path):
    cap = rgb_data['cap']
    frame_count = rgb_data['frame_count']
    every_nth_frame = flags.every

    depth_frames = csv_data['depth_frames']
    depth_confidence_frames = csv_data['depth_confidence_frames']
    odometry = csv_data['odometry']
    imu = csv_data['imu']
    location = csv_data['location']
    heading = csv_data['heading']

    time_diff = get_data_sync_params(flags, csv_data)
    print(f"Time difference for synchronization: {time_diff:.6f} seconds")

    subdir = os.path.basename(flags.path)

    for i in range(0, frame_count, every_nth_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {i}")
            continue
        
        rgb_frame_path = os.path.join(output_dir, 'rgb', f"{subdir}_frame_{i:06d}.png")
        # Rotate frame clockwise by 90 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Get the corresponding depth frame
        ## As mentioned, the depth frames are expected to be in the same order as the RGB frames.
        depth_frame_index = i
        if depth_frame_index < len(depth_frames):
            depth_path = depth_frames[depth_frame_index]
        if depth_frame_index < len(depth_confidence_frames):
            depth_confidence_path = depth_confidence_frames[depth_frame_index] if depth_confidence_frames else None
        if depth_path is None or depth_confidence_path is None:
            print(f"Skipping frame {i} due to missing depth or confidence data.")
            continue

        ## Load the depth frame and confidence frame
        depth_image = load_depth_file(depth_path)
        ## Rotate depth image to match RGB frame orientation
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)
        confidence_image = load_confidence(depth_confidence_path)
        ## Rotate confidence image to match RGB frame orientation
        confidence_image = cv2.rotate(confidence_image, cv2.ROTATE_90_CLOCKWISE)
        if depth_image is None or confidence_image is None:
            print(f"Skipping frame {i} due to failed depth or confidence loading.")
            continue

        ## Print debug information
        print(f"Frame {i}: Depth frame loaded from {depth_path}, Confidence frame loaded from {depth_confidence_path}")

        intrinsics = csv_data['intrinsics']
        # Load the corresponding odometry, imu, location, and heading data
        odometry_data = odometry[i] if i < len(odometry) else None
        if odometry_data is None: 
            print(f"Skipping frame {i} due to missing odometry data.")
            continue
        ## Load imu data using the timestamp from odometry
        odometry_timestamp = odometry_data[0]
        imu_data_index = np.abs(imu[:, 0] - odometry_timestamp).argmin()
        imu_data = imu[imu_data_index] if imu_data_index < len(imu) else None
        # Load location data using the timestamp from odometry and time difference
        location_data_index = np.abs(location[:, 0] - (odometry_timestamp + time_diff)).argmin()
        location_data = location[location_data_index] if location_data_index < len(location) else None
        # Load heading data using the timestamp from odometry and time difference
        heading_data_index = np.abs(heading[:, 0] - (odometry_timestamp + time_diff)).argmin()
        heading_data = heading[heading_data_index] if heading_data_index < len(heading) else None

        # Write the image and depth data to the dataset CSV file
        cv2.imwrite(rgb_frame_path, frame)
        print(f"Saved {rgb_frame_path}")

        depth_frame_path = os.path.join(output_dir, 'depth', f"{subdir}_depth_frame_{i:06d}.png")
        cv2.imwrite(depth_frame_path, depth_image)
        print(f"Saved {depth_frame_path}")
        confidence_frame_path = os.path.join(output_dir, 'confidence', f"{subdir}_confidence_frame_{i:06d}.png")
        cv2.imwrite(confidence_frame_path, confidence_image)
        print(f"Saved {confidence_frame_path}")

        # Prepare the data for the CSV file
        csv_data_row = [
            i,  # frame_index
            os.path.join(os.path.basename(os.path.dirname(flags.path)), subdir),  # original_path
            rgb_frame_path,  # rgb_frame_path
            depth_frame_path,  # depth_frame_path
            confidence_frame_path,  # depth_confidence_frame_path
            intrinsics[0, 0], intrinsics[0, 1], intrinsics[0, 2],  # intrinsics_00, 01, 02
            intrinsics[1, 0], intrinsics[1, 1], intrinsics[1, 2],  # intrinsics_10, 11, 12
            intrinsics[2, 0], intrinsics[2, 1], intrinsics[2, 2],  # intrinsics_20, 21, 22
            odometry_data[0],  # odometry_timestamp (NOTE: we skip first column named 'frame')
            odometry_data[2], odometry_data[3], odometry_data[4],  # odometry_x, y, z
            odometry_data[5], odometry_data[6], odometry_data[7], odometry_data[8],  # odometry_qx, qy, qz, qw
            imu_data[0],  # imu_timestamp
            imu_data[1], imu_data[2], imu_data[3],  # a_x, a_y, a_z
            imu_data[4], imu_data[5], imu_data[6],  # alpha_x, alpha_y, alpha_z
            location_data[0],  # location_timestamp
            location_data[1], location_data[2], location_data[3],  # latitude, longitude, altitude
            location_data[4], location_data[5],  # horizontal_accuracy, vertical_accuracy
            location_data[6], location_data[7], location_data[8],  # speed, course, floor_level
            heading_data[0],  # heading_timestamp
            heading_data[1], heading_data[2], heading_data[3]  # magnetic_heading, true_heading, heading_accuracy
        ]
        # Write the data to the CSV file
        with open(dataset_csv_path, 'a') as f:
            f.write(','.join(map(str, csv_data_row)) + '\n')
            print(f"Wrote data for frame {i} to {dataset_csv_path}")
    print(f"Extracted frames and saved to {output_dir}")
    # Release the video capture object
    cap.release()

if __name__ == '__main__':
    flags = read_args()
    subdirs = get_sub_directories(flags)

    print(f"Processing {len(subdirs)} datasets in {flags.path}")

    output_dir = flags.output
    # Create output directories
    create_output_directories(output_dir)
    # Create dataset csv file
    dataset_csv_path = create_dataset_csv(output_dir, flags)

    dir_path = os.path.abspath(flags.path)
    for subdir in subdirs:
        print(f"Processing dataset: {subdir}")
        flags.path = os.path.join(dir_path, subdir)

        # Read RGB data
        rgb_data = read_rgb(flags)

        # Read CSV data
        csv_data = read_csv_data(flags)

        # Extract frames
        extract_frames(flags, rgb_data, csv_data, dataset_csv_path)

# python 