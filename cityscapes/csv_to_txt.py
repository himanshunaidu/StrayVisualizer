import pandas as pd

# Get rgb_frame_path and annotation_frame_path from the dataset.csv file and save them to a txt file
dataset_csv_path = 'dataset.csv'
output_txt_path = 'dataset.txt'

# Read the dataset.csv file
df = pd.read_csv(dataset_csv_path)
# Extract the rgb_frame_path and annotation_frame_path columns
frame_paths = df[['rgb_frame_path', 'annotation_frame_path']].values
# Write the paths to the output txt file
with open(output_txt_path, 'w') as f:
    for rgb_path, ann_path in frame_paths:
        # Remove '/' prefix from paths if present
        rgb_path = rgb_path.lstrip('/')
        ann_path = ann_path.lstrip('/')
        f.write(f"{rgb_path},{ann_path}\n")