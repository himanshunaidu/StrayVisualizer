import json
import pandas as pd
from geopy.distance import geodesic
import os

# Load the sample JSON from a file (replace 'data.json' with your actual file path)
FILE_PATH = 'rainier.json'  # Replace with your actual filename
FILE_TAG = 'rainier'
with open(FILE_PATH, 'r') as f:
    data = json.load(f)

# Create dictionaries to hold nodes by ID and by class
nodes = data['elements']
node_dict = {node['id']: node for node in nodes if node['type'] == 'node'}

# Build list of relevant node types
target_classes = {'sidewalk', 'building', 'traffic signal', 'traffic light', 'pole'}

# Prepare storage for results
location_errors = []
width_errors = []
slope_errors = []

for node in nodes:
    if node['type'] != 'node':
        continue

    tags = node.get('tags', {})
    node_class = tags.get('demo:class', '').lower()
    if node_class not in target_classes:
        continue

    node_id = node['id']
    capture_id = tags.get('demo:captureId')

    # Find anchor node (capture node) with same captureId and class not in target classes
    anchor_node = None
    for candidate in nodes:
        if candidate['type'] != 'node':
            continue
        ct = candidate.get('tags', {})
        if ct.get('demo:captureId') == capture_id and ct.get('demo:class', '').lower() not in target_classes:
            anchor_node = candidate
            break

    if not anchor_node:
        continue  # skip if no anchor found

    # Step 1: Get delta between anchor node original and true location
    anchor_tags = anchor_node.get('tags', {})
    try:
        anchor_true_pos = (anchor_node['lat'], anchor_node['lon'])
        anchor_orig_pos = (float(anchor_tags['demo:originalLatitude']), float(anchor_tags['demo:originalLongitude']))
        delta_lat = anchor_true_pos[0] - anchor_orig_pos[0]
        delta_lon = anchor_true_pos[1] - anchor_orig_pos[1]
    except (KeyError, ValueError):
        continue

    # Step 2: Correct inferred position of current node
    try:
        node_orig_pos = (float(tags['demo:originalLatitude']), float(tags['demo:originalLongitude']))
        corrected_lat = node_orig_pos[0] + delta_lat
        corrected_lon = node_orig_pos[1] + delta_lon
        corrected_pos = (corrected_lat, corrected_lon)
        true_pos = (node['lat'], node['lon'])
        pos_error = geodesic(corrected_pos, true_pos).meters
        location_errors.append({
            'node_id': node_id,
            'demo:class': node_class,
            'captureId': capture_id,
            'pos_error_meters': pos_error
        })
    except (KeyError, ValueError):
        continue

    # Step 3: Width error (only for sidewalks)
    if node_class == 'sidewalk':
        try:
            width_true = float(tags['demo:width'])
            width_calc = float(tags.get('demo:calculatedWidth', width_true))  # if missing, assume no error
            width_diff = abs(width_true - width_calc)
        except ValueError:
            width_diff = 0
        width_errors.append({
            'node_id': node_id,
            'captureId': capture_id,
            'width_error': width_diff
        })

        # Step 4: Slope error
        try:
            slope_true = float(tags['demo:slope'])
            slope_calc = float(tags.get('demo:calculatedSlope', slope_true))  # if missing, assume no error
            slope_diff = abs(slope_true - slope_calc)
        except ValueError:
            slope_diff = 0
        slope_errors.append({
            'node_id': node_id,
            'captureId': capture_id,
            'slope_error': slope_diff
        })

# Convert to DataFrames
df_location = pd.DataFrame(location_errors)
df_width = pd.DataFrame(width_errors)
df_slope = pd.DataFrame(slope_errors)

# Save results to CSV files
df_location.to_csv(f'{FILE_TAG}_location_errors.csv', index=False)
df_width.to_csv(f'{FILE_TAG}_width_errors.csv', index=False)
df_slope.to_csv(f'{FILE_TAG}_slope_errors.csv', index=False)