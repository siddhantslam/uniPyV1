#!/usr/bin/env python3
# filepath: train_door_model.py

"""
Training Script for Door Detection Model

This script fine-tunes a YOLOv8 model for door detection using the doorDataset folder.

Usage:
    python train_door_model.py [--epochs 10] [--batch_size 16] [--img_size 640]
                              [--device ""] [--models_dir ./models]

 # For webcam inference:
    python train_door_model.py --webcam --model_path best.pt
    
    # For iPhone camera feed inference:
    python train_door_model.py --webcam --camera_url "http://192.168.1.172:8888/video" --model_path best2.pt 
    
    # For video file inference:
    python train_door_model.py --video path/to/video.mp4 --model_path best.pt
                              """

import argparse
import os
import json
import time
import cv2
from pathlib import Path

from ultralytics import YOLO


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Door Detection Model Trainer")
    parser.add_argument("--models_dir", default="./models", 
                        help="Directory to save trained models")
    parser.add_argument("--base_model", default="best3.pt", 
                        help="Base YOLOv8 model to fine-tune (default: best3.pt)")
    parser.add_argument("--epochs", type=int, default=10, 
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training (default: 16)")
    parser.add_argument("--img_size", type=int, default=640, 
                        help="Image size for training (default: 640)")
    parser.add_argument("--device", default="", 
                        help="Device to use for training (default: auto-select)")
    parser.add_argument("--data_yaml", type=str, default="", 
                        help="Path to data.yaml file (default: doorDataset/dataset/data.yaml)")
    # Add inference-related arguments
    parser.add_argument("--webcam", action="store_true", 
                        help="Run inference on webcam feed")
    parser.add_argument("--video", type=str, default="", 
                        help="Path to video file for inference")
    parser.add_argument("--camera_url", type=str, default="", 
                        help="URL for IP camera feed")
    parser.add_argument("--model_path", type=str, default="best.pt", 
                        help="Path to model weights for inference")
    return parser.parse_args()


def train_model(
    data_yaml: str,
    output_dir: str,
    model_name: str,
    base_model: str = "best3.pt",
    epochs: int = 10,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = ""
) -> str:
    """
    Train a YOLOv8 model.
    
    Args:
        data_yaml: Path to the data.yaml file
        output_dir: Directory to save the trained model
        model_name: Name for the model output directory
        base_model: Base YOLOv8 model to fine-tune
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size for training
        device: Device to use for training
        
    Returns:
        Path to the trained model weights
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO(base_model)
    
    # Start training
    print(f"Starting training with data: {data_yaml}")
    print(f"Using base model: {base_model}")
    print(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=20,  # Early stopping patience
        device=device,
        project=output_dir,
        name=model_name,
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    # Get the path to the best model weights
    best_weights_path = Path(output_dir) / model_name / "weights" / "best.pt"
    
    print(f"Training complete. Best weights saved to: {best_weights_path}")
    
    return str(best_weights_path)


def update_yaml_paths(yaml_path):
    """
    Update the data.yaml file with absolute paths to prevent path resolution issues.
    
    Args:
        yaml_path: Path to the data.yaml file
    
    Returns:
        Path to the updated data.yaml file
    """
    with open(yaml_path, 'r') as f:
        content = f.read()
    
    # Get the directory containing the YAML file
    base_dir = os.path.dirname(os.path.abspath(yaml_path))
    
    # Create a temporary YAML file with absolute paths
    temp_yaml_path = os.path.join(base_dir, "data_absolute.yaml")
    
    # Replace relative paths with absolute paths
    lines = content.strip().split('\n')
    updated_lines = []
    
    for line in lines:
        if line.startswith('train:') and not line.startswith('train: /'):
            train_path = line.split(':', 1)[1].strip()
            updated_lines.append(f"train: {os.path.abspath(os.path.join(base_dir, train_path))}")
        elif line.startswith('val:') and not line.startswith('val: /'):
            val_path = line.split(':', 1)[1].strip()
            updated_lines.append(f"val: {os.path.abspath(os.path.join(base_dir, val_path))}")
        elif line.startswith('test:') and not line.startswith('test: /'):
            test_path = line.split(':', 1)[1].strip()
            updated_lines.append(f"test: {os.path.abspath(os.path.join(base_dir, test_path))}")
        else:
            updated_lines.append(line)
    
    # Write the updated YAML file
    with open(temp_yaml_path, 'w') as f:
        f.write('\n'.join(updated_lines))
    
    print(f"Created updated YAML file with absolute paths: {temp_yaml_path}")
    return temp_yaml_path


def run_webcam_inference(model_path, camera_url=""):
    """Run inference on webcam or IP camera feed."""
    print(f"Running inference with model: {model_path}")
    
    # Load the model
    model = YOLO(model_path)
    
    # Use the specified camera URL or default webcam
    if camera_url:
        print(f"Using camera feed from: {camera_url}")
        cap = cv2.VideoCapture(camera_url)
    else:
        print("Using default webcam")
        cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera feed.")
        return
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Run inference
        results = model(frame)
        
        # Custom visualization
        # Instead of results[0].plot()
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                
                # Custom logic to determine door type
                door_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if is_interior_door(door_crop):
                    label = f"Interior Door {conf:.2f}"
                    color = (0, 255, 0)  # Green for interior
                else:
                    label = f"Door {conf:.2f}"
                    color = (0, 0, 255)  # Red for regular
                
                # Draw box and label
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Show frame
        cv2.imshow("Door Detection", annotated_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
def is_interior_door(door_crop):
    """
    Custom logic to determine if the detected door is an interior door.
    
    Args:
        door_crop: Cropped image of the detected door
    
    Returns:
        bool: True if it's an interior door, False otherwise
    """
    # Placeholder logic for interior door detection
    # Replace with actual logic as needed
    # For example, check color, texture, etc.
    return True  # Default to True for now

def run_video_inference(model_path, video_path):
    """Run inference on a video file."""
    print(f"Running inference on video: {video_path} with model: {model_path}")
    
    # Load the model
    model = YOLO(model_path)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    # Get video properties for output
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video file
    output_path = os.path.splitext(video_path)[0] + "_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Process frames
    frame_count = 0
    while True:
        # Read a frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame)
        
        # Process results and visualize
        annotated_frame = results[0].plot()
        
        # Write to output file
        out.write(annotated_frame)
        
        # Display the annotated frame (optional)
        cv2.imshow("Door Detection", annotated_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Inference complete. Output saved to: {output_path}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Check if we're running inference
    if args.webcam or args.video:
        model_path = args.model_path
        
        # Ensure model path is absolute
        if not os.path.isabs(model_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, model_path)
        
        if args.webcam:
            run_webcam_inference(model_path, args.camera_url)
        elif args.video:
            run_video_inference(model_path, args.video)
        return 0
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Door dataset path (use absolute path)
    if args.data_yaml:
        door_data_yaml = args.data_yaml
        if not os.path.isabs(door_data_yaml):
            # Convert to absolute path if it's not already
            door_data_yaml = os.path.abspath(door_data_yaml)
    else:
        # Use default path
        door_data_dir = os.path.join(current_dir, "doorDataset/dataset")
        door_data_yaml = os.path.join(door_data_dir, "data.yaml")
    
    # Ensure the dataset file exists
    if not os.path.exists(door_data_yaml):
        print(f"Error: Door dataset file not found at: {door_data_yaml}")
        return 1
    
    # Update the YAML file with absolute paths
    updated_door_data_yaml = update_yaml_paths(door_data_yaml)
    
    # Create models output directory (absolute path)
    models_dir = os.path.abspath(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    # Use absolute path for base model
    base_model = args.base_model
    if not os.path.isabs(base_model):
        base_model = os.path.join(current_dir, base_model)
    
    # Initialize training info dictionary
    training_info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
    }
    
    # Train door detection model
    print("\n===== Training Door Detection Model =====\n")
    door_model_path = train_model(
        updated_door_data_yaml,
        models_dir,
        "door_detection",
        base_model,
        args.epochs,
        args.batch_size,
        args.img_size,
        args.device
    )
    
    # Update training info
    training_info["door_model"] = door_model_path
    training_info["door_data"] = updated_door_data_yaml
    
    # Save training info
    training_info_path = os.path.join(models_dir, "door_training_info.json")
    with open(training_info_path, "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n===== Training Complete =====")
    print(f"Training info saved to: {training_info_path}")
    print("\nYou can now use this model with the inference script or run with --webcam or --video option.")


if __name__ == "__main__":
    main()