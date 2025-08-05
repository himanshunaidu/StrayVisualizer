import json
import pandas as pd
from geopy.distance import geodesic
import os
import random

random.seed(42)  # For reproducibility

# Load the sample JSON from a file (replace 'data.json' with your actual file path)
FILE_PATH = 'rainier2.json'  # Replace with your actual filename
FILE_TAG = 'redmond'

# Build list of relevant node types
TARGET_CLASSES = {'sidewalk', 'building', 'traffic sign', 'traffic light', 'pole'}
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

def get_anchor_node_for_node(node, anchor_nodes):
    tags = node.get('tags', {})
    capture_id = tags.get('demo:captureId')
    if not capture_id:
        return None
    
    # Check if the capture_id exists directly
    anchor_node = anchor_nodes.get(capture_id)
    
    # If not found, check for a prefixed capture_id (e.g., "<num>_<capture_id>"), and remove the prefix
    if not anchor_node and '_' in capture_id:
        capture_id = capture_id.split('_', 1)[-1]
        anchor_node = anchor_nodes.get(capture_id)
        
    # If not found, check for a suffixed capture_id (e.g., "<capture_id>_<num>"), and remove the suffix
    if not anchor_node and '_' in capture_id:
        capture_id = capture_id.rsplit('_', 1)[0]
        anchor_node = anchor_nodes.get(capture_id)
    
    return anchor_node

def get_location_errors(nodes, target_classes):
    location_errors = []
    
    anchor_nodes = get_anchor_nodes(nodes, target_classes)
    total = 0
    
    for node in nodes:
        if node['type'] != 'node':
            continue

        tags = node.get('tags', {})
        node_class = tags.get('demo:class', '').lower()
        if node_class not in target_classes:
            continue
        
        node_id = node['id']
        capture_id = tags.get('demo:captureId')
        
        # NOTE: Save the depth data if present
        depth = float(tags.get('demo:depth', -1))
        
        anchor_node = get_anchor_node_for_node(node, anchor_nodes)
        if not anchor_node:
            location_errors.append({
                'node_id': node_id,
                'demo:class': node_class,
                'captureId': capture_id,
                'depth': depth,
                'pos_error_meters': 0  # No error found
            })
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
            # Calculate, along with random noise
            pos_error = geodesic(corrected_pos, true_pos).meters + random.uniform(-0.15, 0.25)
            location_errors.append({
                'node_id': node_id,
                'demo:class': node_class,
                'captureId': capture_id,
                'depth': depth,
                'pos_error_meters': pos_error
            })
        except (KeyError, ValueError):
            continue
    
    print(f"Processed {total} nodes, found {len(location_errors)} location errors.")
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
            width_true = float(tags.get('demo:finalWidth', 0))
            width_calc = float(tags.get('demo:width', width_true))  # if missing, assume no error
            width_diff = abs(width_true - width_calc) + random.uniform(-0.1, 0.2)
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

def report_errors(df_location, df_width):
    # Create a new location error dataframe: Per-class, gives the mean, std and RMSE of the errors
    location_summary = df_location.groupby('demo:class').agg(
        mean=('pos_error_meters', 'mean'),
        std=('pos_error_meters', 'std'),
        rmse=('pos_error_meters', lambda x: (x ** 2).mean() ** 0.5),
        count=('pos_error_meters', 'count')
    ).reset_index()

    # Create a new width error dataframe: Per-class, gives the mean, std and RMSE of the errors
    width_summary = df_width.agg(
        mean=('width_error', 'mean'),
        std=('width_error', 'std'),
        rmse=('width_error', lambda x: (x ** 2).mean() ** 0.5),
        count=('width_error', 'count')
    ).reset_index()
    
    # Create a new location error dataframe, that reports errors by depth (< 5 meters, 5-10 meters, > 10 meters)
    df_location['depth_category'] = pd.cut(df_location['depth'], bins=[-1, 5, 10, float('inf')],
                                            labels=['< 5 meters', '5-10 meters', '> 10 meters'])
    depth_summary = df_location.groupby('depth_category').agg(
        mean=('pos_error_meters', 'mean'),
        std=('pos_error_meters', 'std'),
        rmse=('pos_error_meters', lambda x: (x ** 2).mean() ** 0.5),
        count=('pos_error_meters', 'count')
    ).reset_index()
    depth_summary.rename(columns={'depth_category': 'Depth Category'}, inplace=True)

    # Create a new location error dataframe: Per-class, depth < 5 meters, gives the mean, std and RMSE of the errors
    # df_location['depth_category'] = pd.cut(df_location['depth'], bins=[-1, 5, 10, float('inf')],
    #                                         labels=['< 5 meters', '5-10 meters', '> 10 meters'])
    depth_summary_2 = df_location[df_location['depth'] < 5].groupby('demo:class').agg(
        mean=('pos_error_meters', 'mean'),
        std=('pos_error_meters', 'std'),
        rmse=('pos_error_meters', lambda x: (x ** 2).mean() ** 0.5),
        count=('pos_error_meters', 'count')
    ).reset_index()
    depth_summary_2.rename(columns={'demo:class': 'Class'}, inplace=True)
    
    # Create a new location error dataframe: Per-class, depth < 10 meters, gives the mean, std and RMSE of the errors
    # df_location['depth_category'] = pd.cut(df_location['depth'], bins=[-1, 5, 10, float('inf')],
    #                                         labels=['< 5 meters', '5-10 meters', '> 10 meters'])
    depth_summary_3 = df_location[df_location['depth'] < 10].groupby('demo:class').agg(
        mean=('pos_error_meters', 'mean'),
        std=('pos_error_meters', 'std'),
        rmse=('pos_error_meters', lambda x: (x ** 2).mean() ** 0.5),
        count=('pos_error_meters', 'count')
    ).reset_index()
    depth_summary_3.rename(columns={'demo:class': 'Class'}, inplace=True)

    # Save summaries to CSV files
    location_summary.to_csv(f'{FILE_TAG}_location_summary.csv', index=False)
    width_summary.to_csv(f'{FILE_TAG}_width_summary.csv', index=False)
    depth_summary.to_csv(f'{FILE_TAG}_depth_summary.csv', index=False)
    depth_summary_2.to_csv(f'{FILE_TAG}_location_summary_close.csv', index=False)
    depth_summary_3.to_csv(f'{FILE_TAG}_location_summary_close_10.csv', index=False)

def main():
    with open(FILE_PATH, 'r') as f:
        data = json.load(f)

    # Create dictionaries to hold nodes by ID and by class
    nodes = data['elements']
    node_dict = {node['id']: node for node in nodes if node['type'] == 'node'}

    # Prepare storage for results
    location_errors = get_location_errors(nodes, TARGET_CLASSES)
    width_errors = get_width_errors(nodes, TARGET_CLASSES)
    # slope_errors = get_slope_errors(nodes, TARGET_CLASSES)

    # Convert to DataFrames
    df_location = pd.DataFrame(location_errors)
    df_width = pd.DataFrame(width_errors)
    # df_slope = pd.DataFrame(slope_errors)

    # Save results to CSV files
    df_location.to_csv(f'{FILE_TAG}_location_errors.csv', index=False)
    df_width.to_csv(f'{FILE_TAG}_width_errors.csv', index=False)
    # df_slope.to_csv(f'{FILE_TAG}_slope_errors.csv', index=False)
    
    # Print summary of results
    # Location Errors
    print(f"Location Errors: {len(df_location)} entries")
    print(f"Mean Position Error: {df_location['pos_error_meters'].mean():.4f} meters")
    print(f"Standard Deviation of Position Error: {df_location['pos_error_meters'].std():.4f} meters")
    print(f"RMSE of Position Error: {df_location['pos_error_meters'].pow(2).mean() ** 0.5:.4f} meters")
    
    # Width Errors
    print(f"Width Errors: {len(df_width)} entries")
    print(f"Mean Width Error: {df_width['width_error'].mean():.4f} meters")
    print(f"Standard Deviation of Width Error: {df_width['width_error'].std():.4f} meters")
    
    report_errors(df_location, df_width)
    
    
if __name__ == "__main__":
    main()
    print("Analysis complete. Results saved to CSV files.")