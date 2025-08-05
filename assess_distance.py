import os
import math
import pandas as pd

DATA_PATH = 'output/2025_07_09_15_00_00/dataset.csv'
DATA_COLUMNS = ['frame_index', 'location_timestamp', 'latitude', 'longitude']
TIMESTAMP_COL = 'location_timestamp'
LOCATION_COLS = ['latitude', 'longitude']
# DATA_PATH = 'archive/bellevue.csv'
# DATA_COLUMNS = ['sys_time', 'lat_deci', 'lon_deci']
# TIMESTAMP_COL = 'sys_time'
# LOCATION_COLS = ['lat_deci', 'lon_deci']

dataset = pd.read_csv(DATA_PATH, usecols=DATA_COLUMNS)
# location_timestamp as float then sort
dataset[TIMESTAMP_COL] = dataset[TIMESTAMP_COL].astype(float)
dataset = dataset.sort_values(by=TIMESTAMP_COL)
dataset = dataset[1350:1650]  # Adjust the slice as needed
print(dataset.head())

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371000 # Radius of earth in meters. Use 3956 for miles. Determines return value units.
    return c * r

def assess_distance(dataset):
    """
    Assess the distance between consecutive points in the dataset.
    """
    distances = []
    for i in range(len(dataset) - 1):
        lon1, lat1 = dataset.iloc[i][LOCATION_COLS]
        lon2, lat2 = dataset.iloc[i + 1][LOCATION_COLS]
        distance = haversine(lon1, lat1, lon2, lat2)
        distances.append(distance)
    
    return distances

distances = assess_distance(dataset)
print(f"distances: {distances[:10]} ... {distances[-10:]}")
print(sum(distances), "m total distance")