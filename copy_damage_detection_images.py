"""
This script copies damage detection images from the individual output folders to a dedicated folder. 
"""
import os
import shutil
import glob

DATA_DIR = "output"
DATASET_INPUT_SUB_DIR_NAME = "subsets/damage_detection"
MAIN_OUTPUT_DIR = os.path.join(DATA_DIR, "damage_detection")
# OUTPUT_SIR_SET_2_DIR = os.path.join(MAIN_OUTPUT_DIR, "set_2")
OUTPUT_SIR_SET_3_DIR = os.path.join(MAIN_OUTPUT_DIR, "set_3")

# SET_2_SUB_FOLDER_NAMES = [
#     '2025_05_28_18_00_00', '2025_05_29_17_30_00', '2025_06_09_11_30_00', 
#     '2025_06_11_11_30_00', '2025_06_15_13_30_00', '2025_06_15_16_00_00', 
#     '2025_06_21_11_30_00', '2025_06_27_7_45_00'
# ]
SET_3_SUB_FOLDER_NAMES = [
    '2025_07_06_16_00_00', '2025_07_09_15_00_00', '2025_07_24_11_00_00', 
    '2025_07_31_16_00_00', '2025_08_17_18_00_00', '2025_08_18_13_00_00', 
    '2025_09_16_16_45_00', '2025_09_23_14_30_00', '2026_01_13_11_00_00', 
    '2026_01_30_13_00_00', '2026_01_30_16_00_00', '2026_03_20_11_00_00'
]

def get_sub_folder_names(data_dir):
    """
    Get the names of sub-folders in the data directory, excluding the damage detection folder.
    """
    sub_folder_names = [f.name for f in os.scandir(data_dir) if f.is_dir() and f.name != "damage_detection"]
    sub_folder_names.sort()
    return sub_folder_names

def copy_damage_detection_images(dataset_input_sub_folder_paths, output_dir):
    for dataset_input_sub_folder_path in dataset_input_sub_folder_paths:
        # Get all image files in the damage detection sub-folder
        image_files = glob.glob(os.path.join(dataset_input_sub_folder_path, "*.png"))
        
        # Copy each image file to the output directory
        for image_file in image_files:
            shutil.copy(image_file, output_dir)

if __name__ == "__main__":
    sub_folder_names = SET_3_SUB_FOLDER_NAMES#get_sub_folder_names(DATA_DIR)
    # Get all dataset input sub-folder paths
    dataset_input_sub_folder_paths = [os.path.join(DATA_DIR, sub_folder_name, DATASET_INPUT_SUB_DIR_NAME) for sub_folder_name in sub_folder_names]
    copy_damage_detection_images(dataset_input_sub_folder_paths, OUTPUT_SIR_SET_3_DIR)
