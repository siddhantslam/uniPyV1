#!/usr/bin/env python3
# filepath: uniformat_video_processor.py

"""
Uniformat Element Tracker - MVP

This script processes a video file, detecting Uniformat building elements
at both level 3 and level 4 using fine-tuned YOLOv8 models, and overlays 
bounding boxes and labels on each frame.

Requirements:
    pip install opencv-python ultralytics numpy argparse tqdm
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

# Define default Uniformat colors (RGB format)
# These colors will be used if no mapping file is provided
DEFAULT_UNIFORMAT_COLORS = {
    0: (255, 0, 0),    # B1010 - Floor Construction (Red)
    1: (0, 255, 0),    # B1020 - Roof Construction (Green)
    2: (0, 0, 255),    # B2010 - Exterior Walls (Blue)
    3: (255, 255, 0),  # B2020 - Exterior Windows (Yellow)
    4: (255, 0, 255),  # B2030 - Exterior Doors (Magenta)
    5: (0, 255, 255),  # B3010 - Roof Coverings (Cyan)
    6: (128, 0, 0),    # B3020 - Roof Openings (Maroon)
    7: (0, 128, 0),    # C1010 - Partitions (Dark Green)
    8: (0, 0, 128),    # C1020 - Interior Doors (Navy)
    9: (128, 128, 0),  # C1030 - Fittings (Olive)
    10: (128, 0, 128), # C2010 - Stair Construction (Purple)
    11: (0, 128, 128), # C3030 - Ceiling Finishes (Teal)
    # Add more colors as needed for additional classes
}

# Default Uniformat labels (will be overridden if mapping file is provided)
DEFAULT_UNIFORMAT_LABELS = {
    0: "B1010 - Floor Construction",
    1: "B1020 - Roof Construction",
    2: "B2010 - Exterior Walls",
    3: "B2020 - Exterior Windows",
    4: "B2030 - Exterior Doors",
    5: "B3010 - Roof Coverings",
    6: "B3020 - Roof Openings",
    7: "C1010 - Partitions",
    8: "C1020 - Interior Doors",
    9: "C1030 - Fittings",
    10: "C2010 - Stair Construction",
    11: "C3030 - Ceiling Finishes",
    # Add more labels as needed
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Uniformat Element Tracker")
    parser.add_argument("-i", "--input", required=True, help="Path to input video file")
    parser.add_argument("-o", "--output", required=True, help="Path to output video file")
    parser.add_argument(
        "-m3", 
        "--level3_model", 
        default="models/uniformat_level3/weights/best.pt", 
        help="Path to YOLOv8 model file for Uniformat level 3 classification"
    )
    parser.add_argument(
        "-m4", 
        "--level4_model", 
        default="models/uniformat_level4/weights/best.pt", 
        help="Path to YOLOv8 model file for Uniformat level 4 classification"
    )
    parser.add_argument(
        "--mapping_file",
        default=None,
        help="Path to Uniformat mapping JSON file (generated during training)"
    )
    parser.add_argument(
        "-c", 
        "--confidence", 
        type=float, 
        default=0.5, 
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--level",
        choices=["3", "4", "both"],
        default="both",
        help="Which Uniformat level to display (default: both)"
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Display video during processing"
    )
    return parser.parse_args()


def load_uniformat_mapping(mapping_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load Uniformat mapping information from a JSON file.
    
    Args:
        mapping_file: Path to JSON mapping file (generated during training)
        
    Returns:
        Dictionary containing mapping information
    """
    # Default mapping if no file provided
    mapping = {
        "uniformat_level3_names": {},
        "uniformat_level4_names": {},
        "uniformat_hierarchy": {},
        "level3_to_idx": {},
        "level4_to_idx": {}
    }
    
    if mapping_file and os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                mapping = json.load(f)
            print(f"Loaded Uniformat mapping from {mapping_file}")
        except Exception as e:
            print(f"Error loading mapping file: {str(e)}")
            print("Using default mapping")
    
    return mapping


def initialize_video_capture(video_path: str) -> Tuple[cv2.VideoCapture, int, int, float]:
    """
    Initialize video capture and return video properties.
    
    Args:
        video_path: Path to the input video file
        
    Returns:
        Tuple containing:
            - Video capture object
            - Frame width
            - Frame height
            - Video FPS
    
    Raises:
        FileNotFoundError: If the video file doesn't exist
        RuntimeError: If the video cannot be opened
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    return cap, width, height, fps


def initialize_video_writer(
    output_path: str, width: int, height: int, fps: float
) -> cv2.VideoWriter:
    """
    Initialize video writer object for the output video.
    
    Args:
        output_path: Path to the output video file
        width: Frame width
        height: Frame height
        fps: Video FPS
        
    Returns:
        VideoWriter object
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def process_frame(
    frame: np.ndarray, 
    level3_model: YOLO, 
    level4_model: Optional[YOLO],
    conf_threshold: float,
    level: str = "both",
    mapping: Dict[str, Any] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Process a single frame: detect objects and draw bounding boxes.
    
    Args:
        frame: Input frame as numpy array
        level3_model: YOLOv8 model for level 3 classification
        level4_model: YOLOv8 model for level 4 classification (optional)
        conf_threshold: Confidence threshold for detections
        level: Which Uniformat level to display ("3", "4", or "both")
        mapping: Uniformat mapping dictionary
        
    Returns:
        Tuple containing:
            - Frame with annotations
            - List of detection results
    """
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Run inference with level 3 model (always required)
    level3_results = level3_model(frame, verbose=False)[0]
    
    # Run inference with level 4 model if available and requested
    level4_results = None
    if level4_model is not None and level in ["4", "both"]:
        level4_results = level4_model(frame, verbose=False)[0]
    
    detections = []
    
    # Process level 3 results
    if level3_results is not None:
        for det in level3_results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            
            if conf < conf_threshold:
                continue
                
            # Convert to integers
            x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls])
            
            # Get color for this class
            color = DEFAULT_UNIFORMAT_COLORS.get(cls, (255, 255, 255))  # Default to white if class not in colors
            
            # Get level 3 label
            level3_label = DEFAULT_UNIFORMAT_LABELS.get(cls, f"Class {cls}")
            
            # If we have mapping data, use the more descriptive label
            if mapping and mapping.get("level3_to_idx"):
                # Find class code from index
                for code, idx in mapping["level3_to_idx"].items():
                    if idx == cls:
                        code_name = mapping["uniformat_level3_names"].get(code, "")
                        level3_label = f"{code} - {code_name}"
                        break
            
            # Initialize level 4 data
            level4_cls = None
            level4_label = None
            
            # If level 4 results are available, find matching detection
            # This assumes the level 3 and level 4 models will detect the same objects
            # with similar bounding boxes
            if level4_results is not None and level in ["4", "both"]:
                # Find closest matching box in level 4 results
                best_iou = 0
                best_idx = -1
                
                for i, det4 in enumerate(level4_results.boxes.data.cpu().numpy()):
                    x1_4, y1_4, x2_4, y2_4, conf_4, cls_4 = det4
                    
                    if conf_4 < conf_threshold:
                        continue
                    
                    # Calculate IoU between level 3 and level 4 boxes
                    iou = calculate_iou(
                        [x1, y1, x2, y2], 
                        [int(x1_4), int(y1_4), int(x2_4), int(y2_4)]
                    )
                    
                    if iou > 0.5 and iou > best_iou:  # Minimum IoU of 0.5 to match
                        best_iou = iou
                        best_idx = i
                
                if best_idx >= 0:
                    _, _, _, _, _, cls_4 = level4_results.boxes.data.cpu().numpy()[best_idx]
                    level4_cls = int(cls_4)
                    
                    # Get level 4 label
                    level4_label = f"Class {level4_cls}"
                    
                    # If we have mapping data, use the more descriptive label
                    if mapping and mapping.get("level4_to_idx"):
                        # Find class code from index
                        for code, idx in mapping["level4_to_idx"].items():
                            if idx == level4_cls:
                                code_name = mapping["uniformat_level4_names"].get(code, "")
                                level4_label = f"{code} - {code_name}"
                                break
            
            # BGR for OpenCV
            color_bgr = (color[2], color[1], color[0])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Prepare label text based on which level(s) to display
            if level == "3" or level4_label is None:
                label_text = f"{level3_label}: {conf:.2f}"
            elif level == "4":
                label_text = f"{level4_label}: {conf:.2f}"
            else:  # both
                label_text = f"{level3_label} | {level4_label}: {conf:.2f}"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                annotated_frame, 
                (x1, y1 - text_size[1] - 10), 
                (x1 + text_size[0], y1), 
                color_bgr, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame, 
                label_text, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # Save detection data
            detection_data = {
                "bbox": (x1, y1, x2, y2),
                "confidence": float(conf),
                "level3_class_id": cls,
                "level3_label": level3_label
            }
            
            if level4_cls is not None:
                detection_data.update({
                    "level4_class_id": level4_cls,
                    "level4_label": level4_label
                })
                
            detections.append(detection_data)
    
    return annotated_frame, detections


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First box as [x1, y1, x2, y2]
        box2: Second box as [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2 < x1 or y2 < y1:
        return 0.0  # No overlap
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Check level 3 model file
    if not os.path.exists(args.level3_model):
        raise FileNotFoundError(f"Level 3 model file not found: {args.level3_model}")
    
    # Load Uniformat mapping
    mapping = load_uniformat_mapping(args.mapping_file)
    
    # Load YOLOv8 level 3 model
    print(f"Loading YOLOv8 level 3 model: {args.level3_model}")
    level3_model = YOLO(args.level3_model)
    
    # Load YOLOv8 level 4 model if specified
    level4_model = None
    if args.level4_model and args.level in ["4", "both"]:
        if os.path.exists(args.level4_model):
            print(f"Loading YOLOv8 level 4 model: {args.level4_model}")
            level4_model = YOLO(args.level4_model)
        else:
            print(f"Warning: Level 4 model file not found: {args.level4_model}")
            print("Continuing with level 3 model only")
    
    try:
        # Initialize video capture
        print(f"Opening input video: {args.input}")
        cap, width, height, fps = initialize_video_capture(args.input)
        
        # Initialize video writer
        print(f"Setting up output video: {args.output}")
        writer = initialize_video_writer(args.output, width, height, fps)
        
        # Variables for statistics
        frame_count = 0
        total_time = 0
        total_detections = 0
        
        # Process video frame by frame
        print("Starting video processing...")
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processing frame {frame_count}...")
            
            # Process frame (detect objects and draw bounding boxes)
            start_time = time.time()
            annotated_frame, detections = process_frame(
                frame, level3_model, level4_model, args.confidence, args.level, mapping
            )
            process_time = time.time() - start_time
            
            total_time += process_time
            total_detections += len(detections)
            
            # Write to output video
            writer.write(annotated_frame)
            
            # Display if requested
            if args.display:
                cv2.imshow("Uniformat Element Tracker", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Release resources
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()
        
        # Print statistics
        avg_time = total_time / frame_count if frame_count > 0 else 0
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        avg_detections = total_detections / frame_count if frame_count > 0 else 0
        
        print("\nProcessing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Average processing time: {avg_time:.4f} seconds per frame")
        print(f"Average processing speed: {avg_fps:.2f} FPS")
        print(f"Average detections per frame: {avg_detections:.2f}")
        print(f"Output video saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)