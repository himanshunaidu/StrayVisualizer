import json
import pandas as pd
from geopy.distance import geodesic
import os

# Load the sample JSON from a file (replace 'data.json' with your actual file path)
FILE_PATH = 'rainier.json'  # Replace with your actual filename
FILE_TAG = 'rainier'

# Build list of relevant node types
TARGET_CLASSES = {'sidewalk', 'building', 'traffic signal', 'traffic light', 'pole'}
ANCHOR_AMENITY = 'hospital'

def get_anchor_nodes(nodes, target_classes):
    anchor_nodes = {}
    for node in nodes:
        if node['type'] != 'node':
            continue
        tags = node.get('tags', {})
        amenity = tags.get('amenity', '').lower()
        capture_id = tags.get('demo:captureId')
        if amenity == ANCHOR_AMENITY:
            anchor_nodes[capture_id] = node
    return anchor_nodes

def get_location_errors(nodes, target_classes):
    location_errors = []
    
    anchor_nodes = get_anchor_nodes(nodes, target_classes)
    
    for node in nodes:
        if node['type'] != 'node':
            continue

        tags = node.get('tags', {})
        node_class = tags.get('demo:class', '').lower()
        if node_class not in target_classes:
            continue
        
        node_id = node['id']
        capture_id = tags.get('demo:captureId')
        
        anchor_node = anchor_nodes.get(capture_id)
        if not anchor_node:
            continue
        
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
        
    return location_errors

def get_width_errors(nodes, target_classes):
    width_errors = []
    
    for node in nodes:
        if node['type'] != 'node':
            continue

        tags = node.get('tags', {})
        node_class = tags.get('demo:class', '').lower()
        if node_class != 'sidewalk':
            continue
        
        node_id = node['id']
        capture_id = tags.get('demo:captureId')
        try:
            width_true = float(tags.get('demo:width', 0))
            width_calc = float(tags.get('demo:calculatedWidth', width_true))  # if missing, assume no error
            width_diff = abs(width_true - width_calc)
        except ValueError:
            width_diff = 0
        width_errors.append({
            'node_id': node_id,
            'captureId': capture_id,
            'width_error': width_diff
        })
    
    return width_errors

def get_slope_errors(nodes, target_classes):
    slope_errors = []
    
    for node in nodes:
        if node['type'] != 'node':
            continue

        tags = node.get('tags', {})
        node_class = tags.get('demo:class', '').lower
        if node_class != 'sidewalk':
            continue

        node_id = node['id']
        capture_id = tags.get('demo:captureId')

        # Step 4: Slope error
        try:
            slope_true = float(tags.get('demo:slope', 0))
            slope_calc = float(tags.get('demo:calculatedSlope', slope_true))  # if missing, assume no error
            slope_diff = abs(slope_true - slope_calc)
        except ValueError:
            slope_diff = 0
        slope_errors.append({
            'node_id': node_id,
            'captureId': capture_id,
            'slope_error': slope_diff
        })
    return slope_errors

def main():
    with open(FILE_PATH, 'r') as f:
        data = json.load(f)

    # Create dictionaries to hold nodes by ID and by class
    nodes = data['elements']
    node_dict = {node['id']: node for node in nodes if node['type'] == 'node'}

    # Prepare storage for results
    location_errors = get_location_errors(nodes, TARGET_CLASSES)
    width_errors = get_width_errors(nodes, TARGET_CLASSES)
    slope_errors = get_slope_errors(nodes, TARGET_CLASSES)

    # Convert to DataFrames
    df_location = pd.DataFrame(location_errors)
    df_width = pd.DataFrame(width_errors)
    df_slope = pd.DataFrame(slope_errors)

    # Save results to CSV files
    df_location.to_csv(f'{FILE_TAG}_location_errors.csv', index=False)
    df_width.to_csv(f'{FILE_TAG}_width_errors.csv', index=False)
    df_slope.to_csv(f'{FILE_TAG}_slope_errors.csv', index=False)
    
if __name__ == "__main__":
    main()
    print("Analysis complete. Results saved to CSV files.")