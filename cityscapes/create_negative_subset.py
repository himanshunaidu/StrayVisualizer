"""
Looks into the the dataset, all the subsets of the dataset, and creates a new subset folder of images that do not exist in the subsets.
"""
import os
import glob
import shutil

PATHS = ['output/2025_07_09_15_00_00', 'output/2025_05_29_17_30_00', 'output/2025_06_11_11_30_00', 'output/2025_07_06_16_00_00', 'output/2025_06_09_11_30_00', 'output/2025_06_15_13_30_00', 'output/2025_06_21_11_30_00', 'output/2025_06_15_16_00_00', 'output/2025_05_28_18_00_00', 'output/2025_06_27_7_45_00']
MAIN_FOLDER = 'rgb'
SUBSET_FOLDER = 'subsets'
NEGATIVE_FOLDER = 'negative_subset'

def create_negative_subset(path):
    """
    Create a negative subset of images that do not exist in the given path.
    """
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        return
    
    # Get all files in the directory
    all_files = set(os.listdir(os.path.join(path, MAIN_FOLDER)))
    print(list(all_files)[:10], "...")  # Print first 10 files for debugging
    
    # Get all subset directories
    subset_dirs = glob.glob(os.path.join(path, SUBSET_FOLDER, '*'))
    subset_files = set()
    for subset_dir in subset_dirs:
        if os.path.isdir(subset_dir):
            files = os.listdir(subset_dir)
            subset_files.update(files)
    print(list(subset_files)[:10], "...")  # Print first 10 subset files for debugging
    
    # Create a new directory for negative subset
    negative_subset_path = os.path.join(path, NEGATIVE_FOLDER)
    os.makedirs(negative_subset_path, exist_ok=True)
    
    # Iterate through all files and copy those that are not in the subsets
    for file in all_files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Assuming image files
            source_file = os.path.join(path, MAIN_FOLDER, file)
            dest_file = os.path.join(negative_subset_path, file)
            if file not in subset_files:
                print(f"Copying {file} to {negative_subset_path}")
                shutil.copy(source_file, dest_file)
            # else:
            #     print(f"Skipping {file}, already in subsets.")

# For each path, perform the negative subset creation
for path in PATHS:
    create_negative_subset(path)
    print(f"Negative subset created for {path}.")
    print("\n\n")