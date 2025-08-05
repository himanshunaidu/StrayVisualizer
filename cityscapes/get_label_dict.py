"""
This script reads the label_colors.txt file and creates a dictionary mapping labels ids to their corresponding label names.
(Label id is the index of the label in the file)
Example line: 255 255 0 curb ramp
"""
import os
import json

def get_label_dict(label_colors_file):
    label_dict = {}
    with open(label_colors_file, 'r') as f:
        for idx, line in enumerate(f):
            label_details = line.strip().split()
            label_name = ""
            if len(label_details) < 4:
                label_name = " ".join(label_details[1:])
            else:
                # The first three elements are RGB values, the rest is the label name
                label_name = " ".join(label_details[3:])
            label_dict[idx] = label_name
    return label_dict

def main():
    label_colors_file = 'label_colors.txt'
    label_mapping_dict_file = 'label_mapping_dict.json'
    if not os.path.exists(label_colors_file):
        print(f"Label colors file '{label_colors_file}' does not exist.")
        return

    label_dict = get_label_dict(label_colors_file)

    # Save the label dictionary to a JSON file
    with open(label_mapping_dict_file, 'w') as f:
        json.dump(label_dict, f, indent=4)

    print(f"Label mapping dictionary saved to '{label_mapping_dict_file}'")
    
    # Print the label dictionary
    for label_id, label_name in label_dict.items():
        print(f"{label_id}: {label_name}")

if __name__ == "__main__":
    main()
    print("Label dictionary creation complete.")