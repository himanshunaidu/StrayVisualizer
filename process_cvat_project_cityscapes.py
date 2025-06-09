import os
import numpy as np
np.float = np.float64
np.int = np.int_
from argparse import ArgumentParser
from PIL import Image
import cv2
import platform
import shutil

description = """
This script processes annotations from a CVAT project (exported in Cityscapes format).
It uses the file names to match original images (obtained from StrayScanner) with their corresponding annotations 
and saves the original images with their annotations (Cityscapes format).
"""

usage = """
Basic usage: python process_group.py <path-to-dataset-folder>
"""

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('--cvat-path', type=str, help="Path to StrayScanner dataset to process.")
    parser.add_argument('--image-paths', type=str, nargs='+',
                        help="Path to the images folder.")
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

def save_image_with_annotation(image_path, annotation_path, output_path):
    image_list = os.listdir(image_path)
    annotation_list = os.listdir(annotation_path)
    annotation_list = [img for img in annotation_list if img.endswith('_labelIds.png')]

    for annotation_file in annotation_list:
        annotation_image_file = annotation_file.replace('_gtFine_labelIds.png', '.jpg')

        if annotation_image_file not in image_list:
            # print(f"Image file {annotation_image_file} not found in {image_path}. Skipping annotation {annotation_file}.")
            continue

        # Move the image file to the output path
        image_file_path = os.path.join(image_path, annotation_image_file)
        annotation_image_file_path = os.path.join(output_path, annotation_image_file)
        annotation_image_file_path = annotation_image_file_path.replace('.jpg', '_leftImg8bit.png')
        print(f"Copying {image_file_path} to {annotation_image_file_path}")
        convert_jpg_to_png(image_file_path, annotation_image_file_path)

        # shutil.copy(image_file_path, annotation_image_file_path)
        # print(f"Copied {annotation_image_file} to {output_path}")


if __name__ == '__main__':
    flags = read_args()

    flags.annotation_path = os.path.join(flags.cvat_path, 'gtFine', 'default')
    flags.output_path = os.path.join(flags.cvat_path, 'leftImg8bit', 'default')
    if not os.path.exists(flags.output_path):
        os.makedirs(flags.output_path)

    for image_path in flags.image_paths:
        if not os.path.exists(image_path):
            print(f"Image path {image_path} does not exist. Skipping.")
            continue

        print(f"Processing images in {image_path} with annotations from {flags.annotation_path}")
        save_image_with_annotation(image_path, flags.annotation_path, flags.output_path)
