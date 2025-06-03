import os
import pandas as pd
import math

main_dir = "/Users/himanshu/Desktop/Taskar Center for Accessible Technology/TCAT Python/Datasets/StrayVisualizer/data/2025_05_28_18_00_00"

def process_sub_dir(sub_dir):
    heading_df = pd.read_csv(os.path.join(main_dir, sub_dir, 'heading.csv'))
    imu_df = pd.read_csv(os.path.join(main_dir, sub_dir, 'imu.csv'))

    first_row_timestamp_diff = heading_df['timestamp'].iloc[0] - imu_df['timestamp'].iloc[0]
    middle_row_timestamp_diff = heading_df['timestamp'].iloc[len(heading_df) // 2] - imu_df['timestamp'].iloc[len(imu_df) // 2]
    last_row_timestamp_diff = heading_df['timestamp'].iloc[-1] - imu_df['timestamp'].iloc[-1]
    # print(f"First row timestamp difference: {first_row_timestamp_diff}")
    # print(f"Last row timestamp difference: {last_row_timestamp_diff}")
    diff_variance = math.sqrt((first_row_timestamp_diff - middle_row_timestamp_diff) ** 2 + (last_row_timestamp_diff - middle_row_timestamp_diff) ** 2)
    print(f"Timestamp difference variance: {diff_variance}")

    scale_factor = imu_df['timestamp'].iloc[-1] / heading_df['timestamp'].iloc[-1]
    diff_scale_factor = (heading_df['timestamp'].iloc[-1] - heading_df['timestamp'].iloc[0]) / (imu_df['timestamp'].iloc[-1] - imu_df['timestamp'].iloc[0])
    print(f"Scale factor: {scale_factor}, Difference scale factor: {diff_scale_factor}")

def get_sub_dirs(main_dir):
    sub_dirs = []
    for item in os.listdir(main_dir):
        item_path = os.path.join(main_dir, item)
        if os.path.isdir(item_path) and item != 'processed':
            sub_dirs.append(item)
    return sub_dirs

def main():
    sub_dirs = get_sub_dirs(main_dir)
    print(f"Subdirectories found: {sub_dirs}")
    
    for sub_dir in sub_dirs:
        print(f"Processing subdirectory: {sub_dir}")
        process_sub_dir(sub_dir)
    print("Processing complete.")

if __name__ == "__main__":
    main()