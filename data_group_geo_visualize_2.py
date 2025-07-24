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
This script geographically visualizes datasets collected using the iOSPointMapperDataCollector app.
"""

DATA_PATH = 'archive/redmond.csv'
DATA_COLUMNS = ['sys_time', 'lat_deci', 'lon_deci']
TIMESTAMP_COL = 'sys_time'
LOCATION_COLS = ['lat_deci', 'lon_deci']

def map_viz(data):
    # fig = go.Figure()
    # Seattle, WA
    fig = px.scatter_mapbox(
        data_frame=data,
        lat=data[LOCATION_COLS[0]],
        lon=data[LOCATION_COLS[1]],
        hover_name=data[TIMESTAMP_COL],
        hover_data={
            LOCATION_COLS[0]: True,
            LOCATION_COLS[1]: True,
            # "speed": True,
            # "course": True,
            # "floor_level": True,
            # "location_timestamp": True,
        },
        color_discrete_sequence=['red'],
        title="Mapped Trajectory with Heading",
        center={"lat": data[LOCATION_COLS[0]].mean(), "lon": data[LOCATION_COLS[1]].mean()},
        zoom=14,
        mapbox_style="open-street-map",
    )

    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})

    # fig.show()
    fig.write_html(os.path.join(os.path.dirname(DATA_PATH), "map.html"))
    # print(os.path.join(data_path, "map.html"))

if __name__ == "__main__":

    data = None
    if not os.path.exists(DATA_PATH):
        print(f"Data path {DATA_PATH} does not exist.")
        exit(-1)
    
    try:
        data = pd.read_csv(DATA_PATH, usecols=DATA_COLUMNS)
    except Exception as e:
        print(f"Error processing {DATA_PATH}: {e}")
        exit(-1)
    
    # print(data.head())
    # print(data.columns)
    # exit(-1)

    if data is not None:
        map_viz(data)