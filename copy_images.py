#!/usr/bin/env python3
# filepath: copy_images.py

"""
Quick script to copy extracted images to the processed data directories
"""

import os
import shutil
from pathlib import Path
import json

# Source directory with the PNG files
temp_extract_dir = Path("./temp_extract")

# Destination directories
level3_train_dir = Path("./processed_data/level3/train/images")
level3_val_dir = Path("./processed_data/level3/val/images")
level3_test_dir = Path("./processed_data/level3/test/images")
level4_train_dir = Path("./processed_data/level4/train/images")
level4_val_dir = Path("./processed_data/level4/val/images")
level4_test_dir = Path("./processed_data/level4/test/images")

# Read the uniformat_mapping.json to get the class mappings
with open("./processed_data/uniformat_mapping.json", "r") as f:
    mapping = json.load(f)

# Get the list of PNG files in the temp_extract directory
png_files = list(temp_extract_dir.glob("*.png"))
print(f"Found {len(png_files)} PNG files in {temp_extract_dir}")

# Split the files into train, val, and test sets (70%, 20%, 10%)
num_files = len(png_files)
num_train = int(num_files * 0.7)
num_val = int(num_files * 0.2)

train_files = png_files[:num_train]
val_files = png_files[num_train:num_train + num_val]
test_files = png_files[num_train + num_val:]

print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

# Create empty label files for each image
def create_empty_label_file(file_path, image_name):
    with open(file_path / f"{image_name.stem}.txt", "w") as f:
        # Empty file indicates no annotations
        pass

# Copy train images and create label files
for i, png_file in enumerate(train_files):
    # Copy to level3
    dest_file_level3 = level3_train_dir / f"{png_file.stem}_{i}.png"
    shutil.copy(png_file, dest_file_level3)
    create_empty_label_file(Path("./processed_data/level3/train/labels"), dest_file_level3)
    
    # Copy to level4
    dest_file_level4 = level4_train_dir / f"{png_file.stem}_{i}.png"
    shutil.copy(png_file, dest_file_level4)
    create_empty_label_file(Path("./processed_data/level4/train/labels"), dest_file_level4)

# Copy val images and create label files
for i, png_file in enumerate(val_files):
    # Copy to level3
    dest_file_level3 = level3_val_dir / f"{png_file.stem}_{i}.png"
    shutil.copy(png_file, dest_file_level3)
    create_empty_label_file(Path("./processed_data/level3/val/labels"), dest_file_level3)
    
    # Copy to level4
    dest_file_level4 = level4_val_dir / f"{png_file.stem}_{i}.png"
    shutil.copy(png_file, dest_file_level4)
    create_empty_label_file(Path("./processed_data/level4/val/labels"), dest_file_level4)

# Copy test images and create label files
for i, png_file in enumerate(test_files):
    # Copy to level3
    dest_file_level3 = level3_test_dir / f"{png_file.stem}_{i}.png"
    shutil.copy(png_file, dest_file_level3)
    create_empty_label_file(Path("./processed_data/level3/test/labels"), dest_file_level3)
    
    # Copy to level4
    dest_file_level4 = level4_test_dir / f"{png_file.stem}_{i}.png"
    shutil.copy(png_file, dest_file_level4)
    create_empty_label_file(Path("./processed_data/level4/test/labels"), dest_file_level4)

print("Images copied successfully!")
print(f"Total images copied: {len(train_files) + len(val_files) + len(test_files)}")