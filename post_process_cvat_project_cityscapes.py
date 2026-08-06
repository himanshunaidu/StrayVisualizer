import os
import numpy as np
np.float = np.float64
np.int = np.int_
import pandas as pd
from argparse import ArgumentParser
from PIL import Image
import cv2
import platform
import shutil
import json

description = """
This script post-processes the already-processed CVAT Cityscapes project data.
It processes the dataset.csv file to a return a new .txt file with only the 'rgb_frame_path' and 'annotation_frame_path'.
It also processes the label_colors.txt file to get the label ids and create a new .json file with label_id and class_name. 
    (The label id is simply the line number of the label-color in the label_colors.txt file.)
"""

usage = """
Basic usage: python post_process_cvat_project_cityscapes.py --cvat-path <path-to-cvat-folder>
"""

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
    'magnetic_heading', 'true_heading', 'heading_accuracy',
    'annotation_frame_path'
]

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('--cvat-path', type=str, help="Path to iOSPointMapperDataCollector dataset to process.")
    return parser.parse_args()


def process_dataset_csv(cvat_path):
    dataset_csv_path = os.path.join(cvat_path, 'dataset.csv')
    if not os.path.exists(dataset_csv_path):
        raise FileNotFoundError(f"Dataset CSV file not found at {dataset_csv_path}")

    df = pd.read_csv(dataset_csv_path)
    
    # Check if required columns are present
    required_columns = ['rgb_frame_path', 'annotation_frame_path']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset CSV.")

    # Create a new DataFrame with only the required columns
    processed_df = df[['rgb_frame_path', 'annotation_frame_path']]
    # For both paths, remove the leading '/' if present
    processed_df['rgb_frame_path'] = processed_df['rgb_frame_path'].apply(lambda x: x.lstrip('/'))
    processed_df['annotation_frame_path'] = processed_df['annotation_frame_path'].apply(lambda x: x.lstrip('/'))
    
    output_txt_path = os.path.join(cvat_path, 'dataset.txt')
    with open(output_txt_path, 'w') as f:
        for _, row in processed_df.iterrows():
            f.write(f"{row['rgb_frame_path']},{row['annotation_frame_path']}\n")

    print(f"Processed dataset saved to {output_txt_path}")

def process_label_colors(cvat_path):
    label_colors_path = os.path.join(cvat_path, 'label_colors.txt')
    if not os.path.exists(label_colors_path):
        raise FileNotFoundError(f"Label colors file not found at {label_colors_path}")

    with open(label_colors_path, 'r') as f:
        lines = f.readlines()

    # label_colors.txt line format: rrr ggg bbb label_name
    label_dict = {}
    for idx, line in enumerate(lines):
        parts = line.strip().split(sep=' ')
        if len(parts) < 4:
            continue  # Skip lines that do not have enough parts
        r, g, b = map(int, parts[:3])
        label_name = ' '.join(parts[3:])
        label_dict[idx] = label_name

    # Save the label dictionary to a JSON file
    output_json_path = os.path.join(cvat_path, 'label_mapping_dict.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(label_dict, json_file, indent=4)

    print(f"Processed label colors saved to {output_json_path}")


def main():
    args = read_args()
    cvat_path = args.cvat_path

    if not os.path.exists(cvat_path):
        raise FileNotFoundError(f"CVAT path does not exist: {cvat_path}")

    # Process the dataset CSV file
    process_dataset_csv(cvat_path)

    # Process the label colors file
    process_label_colors(cvat_path)
    
if __name__ == "__main__":
    main()