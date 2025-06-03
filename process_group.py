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

description = """
This script visualizes a group of datasets collected using the Stray Scanner app.
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

def read_rgb(flags):
    video_path = os.path.join(flags.path, 'rgb.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"RGB video file not found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Total frames in video: {frame_count}, FPS: {fps}")
    return cap, frame_count, fps

def read_csv_data(flags):
    intrinsics = np.loadtxt(os.path.join(flags.path, 'camera_matrix.csv'), delimiter=',')
    odometry = np.loadtxt(os.path.join(flags.path, 'odometry.csv'), delimiter=',', skiprows=1)
    imu = np.loadtxt(os.path.join(flags.path, 'imu.csv'), delimiter=',', skiprows=1)
    location = np.loadtxt(os.path.join(flags.path, 'location.csv'), delimiter=',', skiprows=1)
    heading = np.loadtxt(os.path.join(flags.path, 'heading.csv'), delimiter=',', skiprows=1)

    # It is assumed that even if the number of depth and depth confidence frames is different,
    # the files are named in such a way that they correspond to the same sequence number.
    depth_dir = os.path.join(flags.path, 'depth')
    depth_frames = [os.path.join(depth_dir, p) for p in sorted(os.listdir(depth_dir))]
    depth_frames = [f for f in depth_frames if '.npy' in f or '.png' in f]

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
    }

def load_depth_file(path, confidence=None, filter_level=0):
    if path[-4:] == '.npy':
        depth_mm = np.load(path)
    elif path[-4:] == '.png':
        depth_mm = np.array(Image.open(path))
    depth_m = depth_mm.astype(np.float32) / 1000.0
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    return o3d.geometry.Image(depth_m)

def load_confidence(path):
    return np.array(Image.open(path))