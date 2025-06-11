import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
np.float = np.float64
np.int = np.int_
from argparse import ArgumentParser
from PIL import Image

description = """
This script processes datasets collected using the iOSPointMapperDataCollector app.
"""

usage = """
Basic usage: python data_group_geo_visualize.py --data-paths <path-to-dataset-folder> <path-to-dataset-folder> ...
"""

def read_args():
    parser = ArgumentParser(description=description, usage=usage)
    parser.add_argument('--data-paths', type=str, nargs='+',
                        help="Path to the dataset folder with images and other details.")
    return parser.parse_args()

def read_data(data_path):
    # Read the dataset.csv file
    csv_path = os.path.join(data_path, 'dataset.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV file not found at {csv_path}")
    
    data = pd.read_csv(csv_path)

    return data

def map_viz(data):
    # fig = go.Figure()
    # Seattle, WA
    fig = px.scatter_mapbox(
        data_frame=data,
        lat=data["latitude"],
        lon=data["longitude"],
        hover_name=data["frame_index"],
        hover_data={
            "latitude": True,
            "longitude": True,
            "speed": True,
            "course": True,
            "floor_level": True,
            "location_timestamp": True,
        },
        title="Mapped Trajectory with Heading",
        center={"lat": data["latitude"].mean(), "lon": data["longitude"].mean()},
        zoom=14,
        mapbox_style="carto-positron",
    )

    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

    fig.show()

if __name__ == "__main__":
    flags = read_args()

    data = None
    
    for data_path in flags.data_paths:
        if not os.path.exists(data_path):
            print(f"Data path {data_path} does not exist.")
            continue
        
        try:
            data1 = read_data(data_path)
            if data is None:
                data = data1
            else:
                data = pd.concat([data, data1], ignore_index=True)
        except Exception as e:
            print(f"Error processing {data_path}: {e}")
            continue
    
    # print(data.head())
    # print(data.columns)
    # exit(-1)

    if data is not None:
        map_viz(data)