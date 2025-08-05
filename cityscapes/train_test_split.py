import os
import random

"""
This script processes the dataset.txt file to get a training and testing split.
It reads the dataset.txt file, splits the data into training and testing sets, and saves the results to separate files.
"""

file_path = 'dataset.txt'
train_file_path = 'train.txt'
test_file_path = 'val.txt'

with open(file_path, 'r') as f:
    lines = f.readlines()

random.shuffle(lines)

split_ratio = 0.8
split_index = int(len(lines) * split_ratio)

with open(train_file_path, 'w') as f:
    f.writelines(lines[:split_index])

with open(test_file_path, 'w') as f:
    f.writelines(lines[split_index:])