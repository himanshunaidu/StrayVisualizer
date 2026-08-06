import os
import glob

DIRECTORY_PATH = "output"

direct_subfolders = [f.path for f in os.scandir(DIRECTORY_PATH) if f.is_dir()]
print("Direct subfolders:")
print(" ".join([f"./{f}" for f in direct_subfolders]))