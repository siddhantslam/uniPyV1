#!/usr/bin/env python3
# filepath: uniformat_video_processor.py

"""
Uniformat Element Tracker - MVP

This script processes a video file, detecting Uniformat level 3 building elements
using a fine-tuned YOLOv8 model, and overlays bounding boxes and labels on each frame.

Requirements:
    pip install opencv-python ultralytics numpy argparse
"""

import argparse
import os
import time
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

# Define Uniformat colors (RGB format)
# Each Uniformat level 3 element will have a distinct color for better visualization
UNIFORMAT_COLORS = {
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
    # Add more colors as needed for additional classes
}

# Map class indices to Uniformat codes and descriptions
UNIFORMAT_LABELS = {
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
    # Add more labels as needed
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Uniformat Element Tracker")
    parser.add_argument("-i", "--input", required=True, help="Path to input video file")
    parser.add_argument("-o", "--output", required=True, help="Path to output video file")
    parser.add_argument(
        "-m", 
        "--model", 
        default="models/uniformat_yolov8.pt", 
        help="Path to YOLOv8 model file (default: models/uniformat_yolov8.pt)"
    )
    parser.add_argument(
        "-c", 
        "--confidence", 
        type=float, 
        default=0.5, 
        help="Confidence threshold for detections (default: 0.5)"
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Display video during processing"
    )
    return parser.parse_args()


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
    frame: np.ndarray, model: YOLO, conf_threshold: float
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Process a single frame: detect objects and draw bounding boxes.
    
    Args:
        frame: Input frame as numpy array
        model: YOLOv8 model object
        conf_threshold: Confidence threshold for detections
        
    Returns:
        Tuple containing:
            - Frame with annotations
            - List of detection results
    """
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    
    # Run inference
    results = model(frame, verbose=False)[0]
    detections = []
    
    # Process results
    for det in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        
        if conf < conf_threshold:
            continue
            
        # Convert to integers
        x1, y1, x2, y2, cls = map(int, [x1, y1, x2, y2, cls])
        
        # Get color for this class
        color = UNIFORMAT_COLORS.get(cls, (255, 255, 255))  # Default to white if class not in colors
        
        # BGR for OpenCV
        color_bgr = (color[2], color[1], color[0])
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, 2)
        
        # Get label text
        label = UNIFORMAT_LABELS.get(cls, f"Class {cls}")
        label_text = f"{label}: {conf:.2f}"
        
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
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(conf),
            "class_id": cls,
            "label": label
        })
    
    return annotated_frame, detections


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Check model file
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {args.model}")
    model = YOLO(args.model)
    
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
                frame, model, args.confidence
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