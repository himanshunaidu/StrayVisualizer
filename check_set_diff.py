"""
This script looks at 2 image folders and finds the images of folder 1 present in folder 2.
Then creates a new folder and only adds the images from folder 2 that are not present in folder 1.
"""
import os
import glob
import shutil

SET_1_DIR = "output/damage_detection/old_set_0/images"
SET_2_DIR = "output/damage_detection/set_3"
SET_2_OUTPUT_DIR = "output/damage_detection/set_3_diff"

def get_image_filenames(image_dir):
    """
    Get the filenames of all images in the specified directory.
    """
    image_files = glob.glob(os.path.join(image_dir, "*.png"))
    image_filenames = [os.path.basename(image_file) for image_file in image_files]
    return set(image_filenames)

def find_common_images(set_1_dir, set_2_dir):
    """
    Find the common images between two directories.
    """
    set_1_images = get_image_filenames(set_1_dir)
    set_2_images = get_image_filenames(set_2_dir)
    
    common_images = set_1_images.intersection(set_2_images)
    return common_images

def add_images_to_output_dir(set_1_dir, set_2_dir, output_dir):
    """
    Add images from set_2_dir that are not in set_1_dir to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    common_images = find_common_images(set_1_dir, set_2_dir)
    print(f"Found {len(common_images)} common images between {set_1_dir} and {set_2_dir}.")
    for image_file in glob.glob(os.path.join(set_2_dir, "*.png")):
        if os.path.basename(image_file) not in common_images:
            shutil.copy(image_file, output_dir)
    

if __name__ == "__main__":
    add_images_to_output_dir(SET_1_DIR, SET_2_DIR, SET_2_OUTPUT_DIR)