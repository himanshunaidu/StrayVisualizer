import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
np.float = np.float64
np.int = np.int_
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import skvideo.io

description = """
This script visualizes datasets collected using the Stray Scanner app.
"""

usage = """
Basic usage: python stray_visualize.py <path-to-dataset-folder>
"""

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('path', type=str, help="Path to iOSPointMapperDataCollector dataset to process.")
    parser.add_argument("--every", type=int, default=6, help="Use every nth frame")
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for the video playback")
    return parser.parse_args()

def read_data(flags):
    # Columns: timestamp, latitude, longitude, altitude, horizontal_accuracy, vertical_accuracy, speed, course, floor_level
    loc = pd.read_csv(os.path.join(flags.path, "location.csv"), sep=',')
    # Columns: timestamp, magnetic_heading, true_heading, heading_accuracy
    head = pd.read_csv(os.path.join(flags.path, "heading.csv"), sep=',')

    return { 'location': loc, 'heading': head }

def load_rgb_video(flags):
    video_path = os.path.join(flags.path, 'rgb.mp4')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"RGB video file not found at {video_path}")
    rgb_video = skvideo.io.vread(video_path)
    # Get every nth frame
    print(rgb_video.shape)
    rgb_video = rgb_video[::flags.every]
    print(rgb_video.shape)
    return rgb_video

def merge_data(data, flags, rgb_video_length):
    # Merge video data with closest location and heading data
    loc = data['location']
    head = data['heading']

    # Timestamp is in epoch seconds, convert to datetime
    loc['timestamp'] = pd.to_datetime(loc['timestamp'], unit='s')
    head['timestamp'] = pd.to_datetime(head['timestamp'], unit='s')

    # We know that the first video frame corresponds to the first location and heading
    # We also know the that every nth frame corresponds to the nth second in the video
    first_timestamp = loc['timestamp'].iloc[0]
    rgb_df = pd.DataFrame({
        'timestamp': pd.date_range(start=first_timestamp, periods=rgb_video_length, freq=f"{flags.every/flags.fps}s"),
        'frame': range(rgb_video_length)
    })
    # Merge with nearest location and heading (timestamp)
    rgb_loc_df = rgb_df['timestamp'].apply(
        lambda ts: loc.iloc[(loc['timestamp'] - ts).abs().argsort()[:1]]
    ).apply(lambda x: x.squeeze())
    rgb_head_df = rgb_df['timestamp'].apply(
        lambda ts: head.iloc[(head['timestamp'] - ts).abs().argsort()[:1]]
    ).apply(lambda x: x.squeeze())
    # Merge the dataframes
    rgb_df = pd.concat([rgb_df, rgb_loc_df.drop(columns='timestamp'), rgb_head_df.drop(columns='timestamp')], axis=1)
    # rgb_df['rgb'] = list(rgb_video)
    rgb_df.set_index('frame', inplace=True)
    return rgb_df

def load_depth_data_length(flags):
    depth_dir = os.path.join(flags.path, 'depth')
    if not os.path.exists(depth_dir):
        return None
    depth_data_length = len(os.listdir(depth_dir))
    return depth_data_length

def map_viz(df, flags):
    # df = df[:1000]
    # Create a map visualization of the location data
    import plotly.express as px

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lat=df[" latitude"],
        lon=df[" longitude"],
        mode="markers+lines",
        marker=dict(size=5, color="blue"),
        line=dict(color="blue"),
        name="Trajectory"
    ))
    # Heading arrows
    # scale = 0.0002  # scale of arrow direction
    # for _, row in df.iterrows():
    #     lat, lon = row[" latitude"], row[" longitude"]
    #     heading = np.deg2rad(row[" true_heading"] if row[" true_heading"] >= 0 else row[" magnetic_heading"])
    #     dlat = scale * np.cos(heading)
    #     dlon = scale * np.sin(heading)
    #     fig.add_trace(go.Scattermapbox(
    #         lat=[lat, lat + dlat],
    #         lon=[lon, lon + dlon],
    #         mode="lines",
    #         line=dict(width=2, color="red"),
    #         showlegend=False
    #     ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=df[" latitude"].iloc[0], lon=df[" longitude"].iloc[0]),
            zoom=18
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        title="Mapped Trajectory with Heading"
    )

    fig.show()

def validate(flags):
    if not os.path.exists(os.path.join(flags.path, 'rgb.mp4')):
        absolute_path = os.path.abspath(flags.path)
        print(f"The directory {absolute_path} does not appear to be a directory created by the Stray Scanner app.")
        return False
    return True

def main():
    flags = read_args()
    assert (flags.every <= flags.fps), "The 'every' parameter must be less than or equal to the 'fps' parameter."

    if not validate(flags):
        return
    
    # rgb_video = load_rgb_video(flags)
    depth_data_length = load_depth_data_length(flags)
    print(f"Depth data length: {depth_data_length}")
    data = read_data(flags)
    rgb_df = merge_data(data, flags, depth_data_length)

    print("Merged data:")
    print(rgb_df.shape)
    print(rgb_df.columns)

    map_viz(rgb_df, flags)

if __name__ == "__main__":
    main()

