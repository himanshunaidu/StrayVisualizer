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

description = """
This script processes annotations from a CVAT project (exported in Cityscapes format).
It uses the file names to match original images (obtained from StrayScanner) with their corresponding annotations 
and saves the original images along with other data (Cityscapes format along with a dataset.csv file).
"""

usage = """
Basic usage: python process_group.py <path-to-dataset-folder>
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
    'magnetic_heading', 'true_heading', 'heading_accuracy'
]
ANNOTATION_FRAME_PATH_COLUMN = 'annotation_frame_path'

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('--cvat-path', type=str, help="Path to StrayScanner dataset to process.")
    parser.add_argument('--data-paths', type=str, nargs='+',
                        help="Path to the dataset folder with images and other details.")
    # NOTE: For Cityscapes format, the output of the images would generally be leftImg8bit.
    # parser.add_argument('--output_path', type=str, help="Path to save the processed images with annotations.")
    return parser.parse_args()

def convert_jpg_to_png(input_path, output_path):
        try:
            image = Image.open(input_path)
            image.save(output_path, "png")
            print(f"Converted {input_path} to {output_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {input_path}")
        except Exception as e:
            print(f"An error occurred: {e}")

def get_image_name_from_annotation(annotation_file):
    """
    Extracts the image name from the annotation file name.
    Assumes the annotation file name ends with '_gtFine_labelIds.png'.
    """
    if annotation_file.endswith('_gtFine_labelIds.png'):
        return annotation_file.replace('_gtFine_labelIds.png', '.png')
    else:
        raise ValueError(f"Annotation file {annotation_file} does not match expected format.")

def get_matches_for_annotation_image_file(annotation_image_file, dataset_df) -> pd.Series:
    """
    Finds the row in the dataset DataFrame that corresponds to the given annotation image file.
    """   
    # Find the row in the dataset DataFrame
    matches_df = dataset_df['rgb_frame_path'].str.contains(annotation_image_file, na=False)

    if matches_df.any():
        # Return the first match
        return dataset_df[matches_df].iloc[0]
    else:
        raise ValueError(f"No matching image found for annotation file {annotation_image_file} in dataset.")

def create_new_dataset_csv(output_dir):
    csv_path = os.path.join(output_dir, 'dataset.csv')
    if os.path.exists(csv_path):
        print(f"Dataset CSV file already exists at {csv_path}. Overwriting...")
    else:
        with open(csv_path, 'w') as f:
            new_dataset_columns = [column for column in DATASET_CSV_COLUMNS]
            new_dataset_columns.append(ANNOTATION_FRAME_PATH_COLUMN)
            f.write(','.join(new_dataset_columns) + '\n')
        print(f"Created dataset CSV file at {csv_path}")
    return csv_path

def save_data_for_annotation(data_path, annotation_path, output_path, image_output_path,
                             depth_output_path, depth_confidence_output_path):
    dataset_df = pd.read_csv(os.path.join(data_path, 'dataset.csv'))
    annotation_list = os.listdir(annotation_path)
    annotation_list = [img for img in annotation_list if img.endswith('_labelIds.png')]

    new_dataset_path = create_new_dataset_csv(output_path)

    for annotation_file in annotation_list:
        # Find the annotation image file in the dataset_df
        try:
            annotation_image_file = get_image_name_from_annotation(annotation_file)
        except ValueError as e:
            # print(f"Skipping annotation {annotation_file}: {e}")
            continue
        try:
            match_row = get_matches_for_annotation_image_file(annotation_image_file, dataset_df)
            # print(f"Found match for {annotation_image_file} in dataset. {match_row['rgb_frame_path']}")
        except ValueError as e:
            # print(f"Skipping annotation {annotation_file}: {e}")
            continue

        print(f"Processing annotation file {annotation_file} for image {annotation_image_file}")

        # Move the rgb image file to the output path
        image_data_file_path = os.path.join(data_path, match_row['rgb_frame_path'])
        image_output_file_path = os.path.join(image_output_path, annotation_image_file)
        image_output_file_path = image_output_file_path.replace('.png', '_leftImg8bit.png')
        shutil.copy(image_data_file_path, image_output_file_path)
        print("Copied RGB image to", image_output_file_path)

        # Move the depth image file to the output path
        depth_image_file_path = os.path.join(data_path, match_row['depth_frame_path'])
        depth_output_file_path = os.path.join(depth_output_path, annotation_image_file)
        shutil.copy(depth_image_file_path, depth_output_file_path)

        # Move the depth confidence image file to the output path
        depth_confidence_image_file_path = os.path.join(data_path, match_row['depth_confidence_frame_path'])
        depth_confidence_output_file_path = os.path.join(depth_confidence_output_path, annotation_image_file)
        shutil.copy(depth_confidence_image_file_path, depth_confidence_output_file_path)

        # Add row to the dataset DataFrame
        new_row = []
        for column in DATASET_CSV_COLUMNS:
            if column == 'rgb_frame_path':
                row_image_output_file_path = image_output_file_path
                row_image_output_file_path = row_image_output_file_path.replace(output_path, '')
                new_row.append(row_image_output_file_path)
            elif column == 'depth_frame_path':
                row_depth_output_file_path = depth_output_file_path
                row_depth_output_file_path = row_depth_output_file_path.replace(output_path, '')
                new_row.append(row_depth_output_file_path)
            elif column == 'depth_confidence_frame_path':
                row_depth_confidence_output_file_path = depth_confidence_output_file_path
                row_depth_confidence_output_file_path = row_depth_confidence_output_file_path.replace(output_path, '')
                new_row.append(row_depth_confidence_output_file_path)
            else:
                new_row.append(match_row[column])
        # For the annotation frame path, we use the annotation file path
        new_row.append(os.path.join(annotation_path, annotation_file).replace(output_path, ''))
        with open(new_dataset_path, 'a') as f:
            f.write(','.join(map(str, new_row)) + '\n')
        


if __name__ == '__main__':
    flags = read_args()

    flags.annotation_path = os.path.join(flags.cvat_path, 'gtFine', 'default')
    flags.image_output_path = os.path.join(flags.cvat_path, 'leftImg8bit', 'default')
    if not os.path.exists(flags.image_output_path):
        os.makedirs(flags.image_output_path)
    flags.depth_output_path = os.path.join(flags.cvat_path, 'depth', 'default')
    if not os.path.exists(flags.depth_output_path):
        os.makedirs(flags.depth_output_path)
    flags.depth_confidence_output_path = os.path.join(flags.cvat_path, 'depth_confidence', 'default')
    if not os.path.exists(flags.depth_confidence_output_path):
        os.makedirs(flags.depth_confidence_output_path)

    for data_path in flags.data_paths:
        if not os.path.exists(data_path):
            print(f"Image path {data_path} does not exist. Skipping.")
            continue

        print(f"Processing images in {data_path} with annotations from {flags.annotation_path}")
        save_data_for_annotation(data_path, flags.annotation_path, flags.cvat_path, flags.image_output_path,
                                 flags.depth_output_path, flags.depth_confidence_output_path)
