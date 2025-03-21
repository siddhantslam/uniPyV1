#!/usr/bin/env python3
# filepath: preprocess_data.py

"""
Data Preprocessing Module for Uniformat Classification

This script preprocesses the FloorPlanCAD and ZInD datasets for training a YOLOv8 model
that can detect and classify building elements according to Uniformat codes at both
level 3 and level 4.

Usage:
    python preprocess_data.py --floorplancad_path /path/to/floorplancad 
                              --zind_path /path/to/zind 
                              --output_path /path/to/output
"""

import argparse
import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Define mappings from original dataset labels to Uniformat codes
# Format: original_label: (level3_code, level4_code)
# This mapping should be customized based on your specific dataset labels
UNIFORMAT_MAPPING = {
    # FloorPlanCAD mappings
    "wall": ("B2010", "B2010.10"),  # Exterior Walls, Exterior Wall Construction
    "window": ("B2020", "B2020.10"),  # Exterior Windows, Windows
    "door": ("B2030", "B2030.10"),  # Exterior Doors, Exterior Entrance Doors
    "room_door": ("C1020", "C1020.10"),  # Interior Doors, Interior Doors
    "sliding_door": ("C1020", "C1020.20"),  # Interior Doors, Interior Specialty Doors
    "balcony_door": ("B2030", "B2030.20"),  # Exterior Doors, Exterior Special Doors
    "railing": ("C1030", "C1030.20"),  # Fittings, Compartments and Cubicles
    "column": ("B1010", "B1010.10"),  # Floor Construction, Floor Structural Frame
    "stairs": ("C2010", "C2010.10"),  # Stair Construction, Stair Construction
    "floor": ("B1010", "B1010.20"),  # Floor Construction, Floor Decks and Slabs
    "ceiling": ("C3030", "C3030.10"),  # Ceiling Finishes, Ceiling Finishes
    "roof": ("B1020", "B1020.10"),  # Roof Construction, Roof Structural Frame
    
    # ZInD mappings - assuming ZInD has similar categories
    "door_zind": ("C1020", "C1020.10"),  # Interior Doors, Interior Doors
    "window_zind": ("B2020", "B2020.10"),  # Exterior Windows, Windows
    "wall_zind": ("C1010", "C1010.10"),  # Partitions, Fixed Partitions
}

# Define hierarchy of Uniformat classes (level 3 to level 4 relationship)
UNIFORMAT_HIERARCHY = {
    # Level 3 to Level 4 mappings
    "B1010": ["B1010.10", "B1010.20", "B1010.90"],  # Floor Construction
    "B1020": ["B1020.10", "B1020.20", "B1020.30", "B1020.90"],  # Roof Construction
    "B2010": ["B2010.10", "B2010.20", "B2010.90"],  # Exterior Walls
    "B2020": ["B2020.10", "B2020.20", "B2020.30", "B2020.90"],  # Exterior Windows
    "B2030": ["B2030.10", "B2030.20", "B2030.30", "B2030.90"],  # Exterior Doors
    "B3010": ["B3010.10", "B3010.20", "B3010.90"],  # Roof Coverings
    "B3020": ["B3020.10", "B3020.20", "B3020.90"],  # Roof Openings
    "C1010": ["C1010.10", "C1010.20", "C1010.90"],  # Partitions
    "C1020": ["C1020.10", "C1020.20", "C1020.90"],  # Interior Doors
    "C1030": ["C1030.10", "C1030.20", "C1030.30", "C1030.90"],  # Fittings
    "C2010": ["C2010.10", "C2010.20", "C2010.90"],  # Stair Construction
    "C3030": ["C3030.10", "C3030.20", "C3030.90"],  # Ceiling Finishes
}

# Define readable names for the Uniformat codes
UNIFORMAT_LEVEL3_NAMES = {
    "B1010": "Floor Construction",
    "B1020": "Roof Construction",
    "B2010": "Exterior Walls",
    "B2020": "Exterior Windows",
    "B2030": "Exterior Doors",
    "B3010": "Roof Coverings",
    "B3020": "Roof Openings",
    "C1010": "Partitions",
    "C1020": "Interior Doors",
    "C1030": "Fittings",
    "C2010": "Stair Construction",
    "C3030": "Ceiling Finishes",
}

UNIFORMAT_LEVEL4_NAMES = {
    "B1010.10": "Floor Structural Frame",
    "B1010.20": "Floor Decks and Slabs",
    "B1010.90": "Other Floor Construction",
    "B1020.10": "Roof Structural Frame",
    "B1020.20": "Roof Decks and Slabs",
    "B1020.30": "Canopy Construction",
    "B1020.90": "Other Roof Construction",
    "B2010.10": "Exterior Wall Construction",
    "B2010.20": "Exterior Wall Finishes",
    "B2010.90": "Other Exterior Wall Construction",
    "B2020.10": "Windows",
    "B2020.20": "Curtain Walls",
    "B2020.30": "Storefronts",
    "B2020.90": "Other Exterior Windows",
    "B2030.10": "Exterior Entrance Doors",
    "B2030.20": "Exterior Special Doors",
    "B2030.30": "Storefronts",
    "B2030.90": "Other Exterior Doors",
    "B3010.10": "Roof Finishes",
    "B3010.20": "Traffic Coatings",
    "B3010.90": "Other Roof Coverings",
    "B3020.10": "Roof Windows and Skylights",
    "B3020.20": "Roof Hatches",
    "B3020.90": "Other Roof Openings",
    "C1010.10": "Fixed Partitions",
    "C1010.20": "Demountable Partitions",
    "C1010.90": "Other Partitions",
    "C1020.10": "Interior Doors",
    "C1020.20": "Interior Specialty Doors",
    "C1020.90": "Other Interior Doors",
    "C1030.10": "Visual Display Units",
    "C1030.20": "Compartments and Cubicles",
    "C1030.30": "Storage Specialties",
    "C1030.90": "Other Fittings",
    "C2010.10": "Stair Construction",
    "C2010.20": "Stair Finishes",
    "C2010.90": "Other Stair Construction",
    "C3030.10": "Ceiling Finishes",
    "C3030.20": "Special Ceiling Finishes",
    "C3030.90": "Other Ceiling Finishes",
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Uniformat Dataset Preprocessor")
    parser.add_argument("--floorplancad_path", required=True, 
                        help="Path to FloorPlanCAD dataset")
    parser.add_argument("--zind_path", required=True, 
                        help="Path to ZInD dataset")
    parser.add_argument("--output_path", required=True, 
                        help="Path to save processed datasets")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def process_floorplancad_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Process the FloorPlanCAD dataset.
    
    Args:
        dataset_path: Path to the FloorPlanCAD dataset
        
    Returns:
        List of processed annotations with images and bounding boxes
    """
    print("Processing FloorPlanCAD dataset...")
    processed_data = []
    
    # Assuming FloorPlanCAD has a specific structure with annotations in JSON
    # Modify this according to the actual dataset structure
    dataset_path = Path(dataset_path)
    
    # Recursively find all image files
    image_files = list(dataset_path.glob("**/*.png")) + list(dataset_path.glob("**/*.jpg"))
    
    for image_file in tqdm(image_files):
        # Assuming annotations are in a JSON file with the same name
        annotation_file = image_file.with_suffix('.json')
        
        if not annotation_file.exists():
            continue
            
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                
            # Assuming the annotation format contains objects with labels and bboxes
            # Modify according to actual format
            bboxes = []
            for obj in annotations.get('objects', []):
                label = obj.get('label', '')
                bbox = obj.get('bbox', [])  # [x, y, width, height] or [x1, y1, x2, y2]
                
                if not label or not bbox or label not in UNIFORMAT_MAPPING:
                    continue
                    
                # Convert bbox to YOLO format if needed
                # YOLO format: [x_center/img_width, y_center/img_height, width/img_width, height/img_height]
                # This conversion depends on the original format
                
                # Map original label to Uniformat codes
                level3_code, level4_code = UNIFORMAT_MAPPING[label]
                
                bboxes.append({
                    'original_label': label,
                    'level3_code': level3_code,
                    'level4_code': level4_code,
                    'bbox': bbox
                })
                
            if bboxes:
                processed_data.append({
                    'image_path': str(image_file),
                    'bboxes': bboxes
                })
                
        except Exception as e:
            print(f"Error processing {annotation_file}: {str(e)}")
            
    return processed_data


def process_zind_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """
    Process the ZInD dataset.
    
    Args:
        dataset_path: Path to the ZInD dataset
        
    Returns:
        List of processed annotations with images and bounding boxes
    """
    print("Processing ZInD dataset...")
    processed_data = []
    
    # Assuming ZInD has a specific structure
    # Modify this according to the actual dataset structure
    dataset_path = Path(dataset_path)
    
    # Recursively find all image files
    image_files = list(dataset_path.glob("**/*.png")) + list(dataset_path.glob("**/*.jpg"))
    
    for image_file in tqdm(image_files):
        # Assuming annotations are in a JSON file with the same name
        annotation_file = image_file.with_suffix('.json')
        
        if not annotation_file.exists():
            continue
            
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
                
            # Modify according to actual ZInD format
            bboxes = []
            for obj in annotations.get('objects', []):
                # Add "_zind" suffix to distinguish from FloorPlanCAD labels
                label = obj.get('label', '') + "_zind"
                bbox = obj.get('bbox', [])
                
                if not label or not bbox or label not in UNIFORMAT_MAPPING:
                    continue
                    
                # Map original label to Uniformat codes
                level3_code, level4_code = UNIFORMAT_MAPPING[label]
                
                bboxes.append({
                    'original_label': label,
                    'level3_code': level3_code,
                    'level4_code': level4_code,
                    'bbox': bbox
                })
                
            if bboxes:
                processed_data.append({
                    'image_path': str(image_file),
                    'bboxes': bboxes
                })
                
        except Exception as e:
            print(f"Error processing {annotation_file}: {str(e)}")
            
    return processed_data


def create_yolo_dataset(
    processed_data: List[Dict[str, Any]],
    output_path: str,
    val_split: float = 0.2,
    test_split: float = 0.1,
    seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Create YOLO dataset structure with both level 3 and level 4 annotations.
    
    Args:
        processed_data: List of processed annotations
        output_path: Path to save the YOLO dataset
        val_split: Validation split ratio
        test_split: Test split ratio
        seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - Dictionary mapping level 3 codes to class indices
            - Dictionary mapping level 4 codes to class indices
    """
    print("Creating YOLO dataset structure...")
    
    # Create output directories
    output_path = Path(output_path)
    
    # Create different output structures for level 3 and level 4
    level3_path = output_path / "level3"
    level4_path = output_path / "level4"
    
    for base_path in [level3_path, level4_path]:
        for split in ['train', 'val', 'test']:
            # Create directories for images and labels
            (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle data for random splitting
    random.shuffle(processed_data)
    
    # Calculate split indices
    num_samples = len(processed_data)
    num_val = int(num_samples * val_split)
    num_test = int(num_samples * test_split)
    num_train = num_samples - num_val - num_test
    
    # Split data
    train_data = processed_data[:num_train]
    val_data = processed_data[num_train:num_train+num_val]
    test_data = processed_data[num_train+num_val:]
    
    # Collect unique Uniformat codes
    level3_codes = set()
    level4_codes = set()
    
    for item in processed_data:
        for bbox in item['bboxes']:
            level3_codes.add(bbox['level3_code'])
            level4_codes.add(bbox['level4_code'])
            
    # Create mapping dictionaries for class indices
    level3_to_idx = {code: idx for idx, code in enumerate(sorted(level3_codes))}
    level4_to_idx = {code: idx for idx, code in enumerate(sorted(level4_codes))}
    
    # Process splits
    for split_name, split_data in [
        ('train', train_data), 
        ('val', val_data), 
        ('test', test_data)
    ]:
        for idx, item in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            image_path = Path(item['image_path'])
            image_name = image_path.name
            
            # Read image to get dimensions
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Create level 3 annotations
            level3_labels = []
            for bbox in item['bboxes']:
                x, y, w, h = bbox['bbox']  # Assuming [x, y, width, height] format
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Get class index for level 3
                class_idx = level3_to_idx[bbox['level3_code']]
                
                # YOLO format: class_idx center_x center_y width height
                level3_labels.append(f"{class_idx} {x_center} {y_center} {width} {height}")
                
            # Create level 4 annotations
            level4_labels = []
            for bbox in item['bboxes']:
                x, y, w, h = bbox['bbox']
                
                # Convert to YOLO format (normalized center coordinates)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Get class index for level 4
                class_idx = level4_to_idx[bbox['level4_code']]
                
                # YOLO format: class_idx center_x center_y width height
                level4_labels.append(f"{class_idx} {x_center} {y_center} {width} {height}")
                
            # Copy image to both level directories
            dest_image_level3 = level3_path / split_name / 'images' / image_name
            dest_image_level4 = level4_path / split_name / 'images' / image_name
            
            shutil.copy(str(image_path), str(dest_image_level3))
            shutil.copy(str(image_path), str(dest_image_level4))
            
            # Write label files
            with open(level3_path / split_name / 'labels' / f"{image_path.stem}.txt", 'w') as f:
                f.write('\n'.join(level3_labels))
                
            with open(level4_path / split_name / 'labels' / f"{image_path.stem}.txt", 'w') as f:
                f.write('\n'.join(level4_labels))
                
    # Create data.yaml files for both levels
    level3_names = {idx: f"{code} - {UNIFORMAT_LEVEL3_NAMES.get(code, '')}" 
                    for code, idx in level3_to_idx.items()}
    
    level4_names = {idx: f"{code} - {UNIFORMAT_LEVEL4_NAMES.get(code, '')}" 
                    for code, idx in level4_to_idx.items()}
    
    level3_yaml = {
        'path': str(level3_path),
        'train': str(level3_path / 'train'),
        'val': str(level3_path / 'val'),
        'test': str(level3_path / 'test'),
        'nc': len(level3_codes),
        'names': level3_names
    }
    
    level4_yaml = {
        'path': str(level4_path),
        'train': str(level4_path / 'train'),
        'val': str(level4_path / 'val'),
        'test': str(level4_path / 'test'),
        'nc': len(level4_codes),
        'names': level4_names
    }
    
    with open(level3_path / 'data.yaml', 'w') as f:
        yaml.dump(level3_yaml, f, sort_keys=False)
        
    with open(level4_path / 'data.yaml', 'w') as f:
        yaml.dump(level4_yaml, f, sort_keys=False)
        
    print(f"YOLO dataset created at {output_path}")
    print(f"Level 3 classes: {len(level3_codes)}")
    print(f"Level 4 classes: {len(level4_codes)}")
    
    return level3_to_idx, level4_to_idx


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Process datasets
    floorplancad_data = process_floorplancad_dataset(args.floorplancad_path)
    zind_data = process_zind_dataset(args.zind_path)
    
    # Combine datasets
    combined_data = floorplancad_data + zind_data
    
    # Create YOLO dataset structure
    level3_to_idx, level4_to_idx = create_yolo_dataset(
        combined_data,
        args.output_path,
        args.val_split,
        args.test_split,
        args.seed
    )
    
    # Save mapping dictionaries for later use in the inference script
    mapping_data = {
        'level3_to_idx': level3_to_idx,
        'level4_to_idx': level4_to_idx,
        'uniformat_level3_names': UNIFORMAT_LEVEL3_NAMES,
        'uniformat_level4_names': UNIFORMAT_LEVEL4_NAMES,
        'uniformat_hierarchy': UNIFORMAT_HIERARCHY
    }
    
    with open(Path(args.output_path) / 'uniformat_mapping.json', 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print("Data preprocessing complete!")


if __name__ == "__main__":
    main()