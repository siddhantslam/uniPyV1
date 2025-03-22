#!/usr/bin/env python3
# filepath: preprocess_data.py

"""
Data Preprocessing Module for Uniformat Classification

This script preprocesses the FloorPlanCAD dataset for training a YOLOv8 model
that can detect and classify building elements according to Uniformat codes at both
level 3 and level 4.

Usage:
    python preprocess_data.py --floorplancad_path /path/to/floorplancad
                              --output_path /path/to/output
"""

import argparse
import os
import json
import shutil
import random
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Define mappings from original dataset labels to Uniformat codes
# Format: original_label: (level3_code, level4_code)
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
}

# Additional mapping for SVG elements to labels
SVG_ELEMENT_MAPPING = {
    "rect": ["wall", "window", "door"],  # These can be rectangles in SVG
    "line": ["wall"],  # Walls could be lines
    "path": ["wall", "stairs", "railing"],  # Complex shapes like stairs or railings
    "circle": ["column"],  # Columns might be circles
    "polygon": ["door", "window", "room_door", "sliding_door", "balcony_door"]  # Various door/window types
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
    parser.add_argument("--floorplancad_path", default="./archive", 
                        help="Path to FloorPlanCAD dataset archive (default: ./archive)")
    parser.add_argument("--zind_path", default=None, 
                        help="Path to ZInD dataset (optional)")
    parser.add_argument("--output_path", required=True, 
                        help="Path to save processed datasets")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--test_split", type=float, default=0.1,
                        help="Test split ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


def extract_and_process_floorplancad(archive_path: str) -> List[Dict[str, Any]]:
    """
    Extract and process the FloorPlanCAD dataset from tar.xz archives.
    
    Args:
        archive_path: Path to directory containing FloorPlanCAD archive files
        
    Returns:
        List of processed annotations with images and bounding boxes
    """
    print("Processing FloorPlanCAD dataset...")
    processed_data = []
    
    # Create temporary directory to extract archives
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = Path(archive_path)
        
        # Find all tar.xz archives in the specified path
        archive_files = list(archive_path.glob("*.tar.xz"))
        
        if not archive_files:
            print(f"No tar.xz files found in {archive_path}")
            return processed_data
            
        print(f"Found {len(archive_files)} archive files")
        
        # Extract each archive
        for archive_file in archive_files:
            print(f"Extracting {archive_file}...")
            with tarfile.open(archive_file, "r:xz") as tar:
                tar.extractall(path=temp_path)
        
        # Find all image files (png) and their corresponding SVG files
        png_files = list(temp_path.glob("**/*.png"))
        print(f"Found {len(png_files)} PNG images")
        
        # Process each image file that has a corresponding SVG
        for png_file in tqdm(png_files, desc="Processing images"):
            # Look for corresponding SVG file
            svg_file = png_file.with_suffix('.svg')
            
            if not svg_file.exists():
                continue
                
            try:
                # Get image dimensions
                img = cv2.imread(str(png_file))
                if img is None:
                    print(f"Warning: Could not read image {png_file}")
                    continue
                    
                img_height, img_width = img.shape[:2]
                
                # Process SVG file to extract annotations
                bboxes = extract_annotations_from_svg(svg_file, img_width, img_height)
                
                if bboxes:
                    # Add to processed data
                    processed_data.append({
                        'image_path': str(png_file),
                        'image_height': img_height,
                        'image_width': img_width,
                        'bboxes': bboxes
                    })
                    
            except Exception as e:
                print(f"Error processing {png_file} and {svg_file}: {str(e)}")
    
    print(f"Processed {len(processed_data)} valid images with annotations")
    return processed_data


def extract_annotations_from_svg(svg_file: Path, img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """
    Extract bounding box annotations from an SVG file.
    
    Args:
        svg_file: Path to the SVG file
        img_width: Width of the corresponding image
        img_height: Height of the corresponding image
        
    Returns:
        List of bounding box annotations with labels
    """
    bboxes = []
    
    try:
        # Parse SVG file
        tree = ET.parse(svg_file)
        root = tree.getroot()
        
        # SVG namespace
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Process different SVG elements
        # Rectangles (could be walls, windows, doors)
        for rect in root.findall('.//svg:rect', ns):
            try:
                x = float(rect.get('x', 0))
                y = float(rect.get('y', 0))
                width = float(rect.get('width', 0))
                height = float(rect.get('height', 0))
                
                # Look for class information in the style or class attributes
                style = rect.get('style', '')
                class_attr = rect.get('class', '')
                
                # Try to determine label based on attributes
                label = determine_label_from_attributes(style, class_attr, 'rect')
                
                if label and label in UNIFORMAT_MAPPING:
                    level3_code, level4_code = UNIFORMAT_MAPPING[label]
                    
                    bboxes.append({
                        'original_label': label,
                        'level3_code': level3_code,
                        'level4_code': level4_code,
                        'bbox': [x, y, width, height]  # [x, y, width, height]
                    })
            except Exception as e:
                print(f"Error processing rectangle in {svg_file}: {str(e)}")
        
        # Lines (likely walls)
        for line in root.findall('.//svg:line', ns):
            try:
                x1 = float(line.get('x1', 0))
                y1 = float(line.get('y1', 0))
                x2 = float(line.get('x2', 0))
                y2 = float(line.get('y2', 0))
                
                # Convert line to bbox
                x = min(x1, x2)
                y = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                # Ensure the box has some size
                if width < 1: width = 3
                if height < 1: height = 3
                
                style = line.get('style', '')
                class_attr = line.get('class', '')
                
                label = determine_label_from_attributes(style, class_attr, 'line')
                
                if label and label in UNIFORMAT_MAPPING:
                    level3_code, level4_code = UNIFORMAT_MAPPING[label]
                    
                    bboxes.append({
                        'original_label': label,
                        'level3_code': level3_code,
                        'level4_code': level4_code,
                        'bbox': [x, y, width, height]
                    })
            except Exception as e:
                print(f"Error processing line in {svg_file}: {str(e)}")
        
        # Paths (walls, stairs, railing)
        for path in root.findall('.//svg:path', ns):
            try:
                # Get path data
                d = path.get('d', '')
                
                # Extract points from path data
                points = extract_points_from_path(d)
                
                if points:
                    # Calculate bounding box from points
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    x = min(x_coords)
                    y = min(y_coords)
                    width = max(x_coords) - x
                    height = max(y_coords) - y
                    
                    style = path.get('style', '')
                    class_attr = path.get('class', '')
                    
                    label = determine_label_from_attributes(style, class_attr, 'path')
                    
                    if label and label in UNIFORMAT_MAPPING:
                        level3_code, level4_code = UNIFORMAT_MAPPING[label]
                        
                        bboxes.append({
                            'original_label': label,
                            'level3_code': level3_code,
                            'level4_code': level4_code,
                            'bbox': [x, y, width, height]
                        })
            except Exception as e:
                print(f"Error processing path in {svg_file}: {str(e)}")
        
        # Circles (columns)
        for circle in root.findall('.//svg:circle', ns):
            try:
                cx = float(circle.get('cx', 0))
                cy = float(circle.get('cy', 0))
                r = float(circle.get('r', 0))
                
                # Convert circle to bbox
                x = cx - r
                y = cy - r
                width = 2 * r
                height = 2 * r
                
                style = circle.get('style', '')
                class_attr = circle.get('class', '')
                
                label = determine_label_from_attributes(style, class_attr, 'circle')
                
                if label and label in UNIFORMAT_MAPPING:
                    level3_code, level4_code = UNIFORMAT_MAPPING[label]
                    
                    bboxes.append({
                        'original_label': label,
                        'level3_code': level3_code,
                        'level4_code': level4_code,
                        'bbox': [x, y, width, height]
                    })
            except Exception as e:
                print(f"Error processing circle in {svg_file}: {str(e)}")
        
        # Polygons (various shapes)
        for polygon in root.findall('.//svg:polygon', ns):
            try:
                points_str = polygon.get('points', '')
                point_pairs = points_str.strip().split()
                
                if point_pairs:
                    points = []
                    for pair in point_pairs:
                        x, y = pair.split(',')
                        points.append((float(x), float(y)))
                    
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    x = min(x_coords)
                    y = min(y_coords)
                    width = max(x_coords) - x
                    height = max(y_coords) - y
                    
                    style = polygon.get('style', '')
                    class_attr = polygon.get('class', '')
                    
                    label = determine_label_from_attributes(style, class_attr, 'polygon')
                    
                    if label and label in UNIFORMAT_MAPPING:
                        level3_code, level4_code = UNIFORMAT_MAPPING[label]
                        
                        bboxes.append({
                            'original_label': label,
                            'level3_code': level3_code,
                            'level4_code': level4_code,
                            'bbox': [x, y, width, height]
                        })
            except Exception as e:
                print(f"Error processing polygon in {svg_file}: {str(e)}")
        
        # If no annotations found based on element types, try to extract from CSS classes
        if not bboxes:
            # Elements with class attributes
            for elem in root.findall('.//*[@class]', ns):
                try:
                    class_attr = elem.get('class', '')
                    
                    # Try to infer label from class
                    for potential_label in UNIFORMAT_MAPPING.keys():
                        if potential_label in class_attr.lower():
                            # Calculate bounding box for this element
                            bbox = calculate_element_bbox(elem, ns)
                            
                            if bbox:
                                level3_code, level4_code = UNIFORMAT_MAPPING[potential_label]
                                
                                bboxes.append({
                                    'original_label': potential_label,
                                    'level3_code': level3_code,
                                    'level4_code': level4_code,
                                    'bbox': bbox
                                })
                                break
                except Exception as e:
                    print(f"Error processing element with class in {svg_file}: {str(e)}")
        
        # If still no labels, use a heuristic approach based on element type and position
        if not bboxes:
            # Heuristic: Let's assume typical elements based on their position and size
            # For example, tall narrow rectangles near perimeter are walls
            
            # Find all rectangles
            for rect in root.findall('.//svg:rect', ns):
                try:
                    x = float(rect.get('x', 0))
                    y = float(rect.get('y', 0))
                    width = float(rect.get('width', 0))
                    height = float(rect.get('height', 0))
                    
                    # Heuristic labels based on size and position
                    label = None
                    
                    # Tall and narrow - likely a wall
                    if height > width * 3:
                        label = "wall"
                    # Small and square - possibly a column
                    elif width < img_width * 0.05 and height < img_height * 0.05 and abs(width - height) < 10:
                        label = "column"
                    # Small and rectangular - could be a window
                    elif width < img_width * 0.1 and height < img_height * 0.1:
                        label = "window"
                    # Medium sized - might be a door
                    elif width < img_width * 0.1 and height < img_height * 0.15:
                        label = "door"
                    # Default to wall for other rectangles
                    else:
                        label = "wall"
                    
                    if label and label in UNIFORMAT_MAPPING:
                        level3_code, level4_code = UNIFORMAT_MAPPING[label]
                        
                        bboxes.append({
                            'original_label': label,
                            'level3_code': level3_code,
                            'level4_code': level4_code,
                            'bbox': [x, y, width, height]
                        })
                except Exception as e:
                    print(f"Error processing rectangle with heuristics in {svg_file}: {str(e)}")
        
        # Print debug info if no annotations found
        if not bboxes:
            print(f"Warning: No annotations extracted from {svg_file}")
        
    except Exception as e:
        print(f"Error parsing SVG file {svg_file}: {str(e)}")
    
    return bboxes


def determine_label_from_attributes(style: str, class_attr: str, element_type: str) -> Optional[str]:
    """
    Determine label from SVG element attributes.
    
    Args:
        style: Style attribute content
        class_attr: Class attribute content
        element_type: Type of SVG element
        
    Returns:
        Determined label or None if no label could be inferred
    """
    style_lower = style.lower()
    class_lower = class_attr.lower()
    
    # Check for explicit label mentions in attributes
    for label in UNIFORMAT_MAPPING.keys():
        if label in style_lower or label in class_lower:
            return label
    
    # If no explicit match, try to infer from element type
    if element_type in SVG_ELEMENT_MAPPING:
        potential_labels = SVG_ELEMENT_MAPPING[element_type]
        
        # For simplicity, return the first potential label
        if potential_labels:
            return potential_labels[0]
    
    # Default label based on element type
    default_labels = {
        'rect': 'wall',
        'line': 'wall',
        'path': 'wall',
        'circle': 'column',
        'polygon': 'door'
    }
    
    return default_labels.get(element_type)


def extract_points_from_path(d: str) -> List[Tuple[float, float]]:
    """
    Extract coordinate points from SVG path data.
    
    Args:
        d: SVG path data string
        
    Returns:
        List of (x, y) coordinate points
    """
    points = []
    
    # Basic regex for SVG path commands followed by coordinates
    pattern = r'([A-Za-z])([^A-Za-z]*)'
    
    # Current position
    current_x, current_y = 0, 0
    
    for command, params in re.findall(pattern, d):
        # Extract parameters as a list of floats
        params = [float(p) for p in re.findall(r'[+-]?[0-9]*\.?[0-9]+', params)]
        
        if not params:
            continue
            
        command = command.upper()
        
        # Handle different path commands
        if command == 'M':  # Move to
            # Absolute coordinates
            for i in range(0, len(params), 2):
                if i + 1 < len(params):
                    current_x, current_y = params[i], params[i+1]
                    points.append((current_x, current_y))
                    
        elif command == 'L':  # Line to
            # Absolute coordinates
            for i in range(0, len(params), 2):
                if i + 1 < len(params):
                    current_x, current_y = params[i], params[i+1]
                    points.append((current_x, current_y))
                    
        elif command == 'H':  # Horizontal line
            # Absolute x coordinate
            for x in params:
                current_x = x
                points.append((current_x, current_y))
                
        elif command == 'V':  # Vertical line
            # Absolute y coordinate
            for y in params:
                current_y = y
                points.append((current_x, current_y))
                
        elif command == 'Z':  # Close path
            # No parameters, connects back to first point
            if points:
                points.append(points[0])
    
    return points


def calculate_element_bbox(elem, ns) -> Optional[List[float]]:
    """
    Calculate bounding box for an SVG element.
    
    Args:
        elem: SVG element
        ns: SVG namespace
        
    Returns:
        [x, y, width, height] or None if can't be calculated
    """
    tag = elem.tag.split('}')[-1]  # Get tag without namespace
    
    if tag == 'rect':
        try:
            x = float(elem.get('x', 0))
            y = float(elem.get('y', 0))
            width = float(elem.get('width', 0))
            height = float(elem.get('height', 0))
            return [x, y, width, height]
        except:
            return None
            
    elif tag == 'circle':
        try:
            cx = float(elem.get('cx', 0))
            cy = float(elem.get('cy', 0))
            r = float(elem.get('r', 0))
            return [cx - r, cy - r, 2 * r, 2 * r]
        except:
            return None
            
    elif tag == 'line':
        try:
            x1 = float(elem.get('x1', 0))
            y1 = float(elem.get('y1', 0))
            x2 = float(elem.get('x2', 0))
            y2 = float(elem.get('y2', 0))
            return [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
        except:
            return None
            
    elif tag == 'polygon' or tag == 'polyline':
        try:
            points_str = elem.get('points', '')
            point_pairs = points_str.strip().split()
            
            if point_pairs:
                points = []
                for pair in point_pairs:
                    x, y = pair.split(',')
                    points.append((float(x), float(y)))
                
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                return [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
        except:
            return None
            
    elif tag == 'path':
        try:
            d = elem.get('d', '')
            points = extract_points_from_path(d)
            
            if points:
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                return [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]
        except:
            return None
    
    return None


def process_zind_dataset(dataset_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Process the ZInD dataset if provided.
    
    Args:
        dataset_path: Path to the ZInD dataset (optional)
        
    Returns:
        List of processed annotations with images and bounding boxes
    """
    if not dataset_path:
        print("ZInD dataset path not provided. Skipping ZInD processing.")
        return []
        
    print("Processing ZInD dataset...")
    processed_data = []
    
    # ZInD processing code would go here if needed
    # For now, we're ignoring the ZInD dataset as requested
    
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
            image_name = f"{image_path.stem}_{idx}{image_path.suffix}"  # Add index to avoid duplicates
            
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
                
                # Make sure values are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
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
                
                # Make sure values are within [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
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
            with open(level3_path / split_name / 'labels' / f"{image_name.split('.')[0]}.txt", 'w') as f:
                f.write('\n'.join(level3_labels))
                
            with open(level4_path / split_name / 'labels' / f"{image_name.split('.')[0]}.txt", 'w') as f:
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
    floorplancad_data = extract_and_process_floorplancad(args.floorplancad_path)
    zind_data = process_zind_dataset(args.zind_path)
    
    # Combine datasets (zind_data will be empty if not provided)
    combined_data = floorplancad_data + zind_data
    
    if not combined_data:
        print("Error: No valid data found to process. Please check your dataset paths.")
        return
    
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