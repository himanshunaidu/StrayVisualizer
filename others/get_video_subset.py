import cv2
import os
from argparse import ArgumentParser

description = """
This script extracts a subset of frames from a video dataset collected using the Stray Scanner app.
"""
usage = """
Basic usage: python get_video_subset.py <path-to-dataset-folder>
"""

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str, help="Path to StrayScanner dataset to process.")
    parser.add_argument("--every", type=int, default=60, help="Use every nth frame")
    return parser.parse_args()

def extract_frames(flags):
    video_path = os.path.join(flags.path, 'rgb.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"RGB video file not found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Total frames in video: {frame_count}, FPS: {fps}")

    extracted_frames = []
    for i in range(0, frame_count, flags.every):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        print(f"Reading frame {i}: Frame shape: {frame.shape if ret else 'None'}")
        if ret:
            extracted_frames.append(frame)
            print(f"Extracted frame {i}")
        else:
            print(f"Failed to read frame {i}")

    cap.release()
    return extracted_frames

def save_frames(frames, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved {frame_path}")

def main():
    flags = read_args()
    output_dir = os.path.join(flags.path, 'rgb')

    frames = extract_frames(flags)
    save_frames(frames, output_dir)

if __name__ == '__main__':
    main()