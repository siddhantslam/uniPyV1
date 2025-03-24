#!/usr/bin/env python3
# filepath: train_models.py

"""
Training Script for Uniformat Classification Models and Door Detection Model

This script fine-tunes YOLOv8 models for:
1. Uniformat level 3 and level 4 building element classification 
2. Door detection 

The datasets should be prepared in the appropriate YOLOv8 format.

Usage:
    python train_models.py --level3_data /path/to/level3/data.yaml 
                          --level4_data /path/to/level4/data.yaml
                          --models_dir /path/to/save/models
    
    # Or for door detection:
    python train_models.py --door_data /path/to/door/data.yaml
                          --models_dir /path/to/save/models
                          
    # For webcam inference:
    python train_models.py --webcam --model_path best.pt
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
    parser = argparse.ArgumentParser(description="Model Trainer for Building Elements")
    parser.add_argument("--level3_data", 
                        help="Path to level 3 data.yaml file")
    parser.add_argument("--level4_data", 
                        help="Path to level 4 data.yaml file")
    parser.add_argument("--door_data",
                        help="Path to door dataset data.yaml file")
    parser.add_argument("--models_dir", 
                        help="Directory to save trained models")
    parser.add_argument("--base_model", default="yolov8n.pt", 
                        help="Base YOLOv8 model to fine-tune (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training (default: 16)")
    parser.add_argument("--img_size", type=int, default=640, 
                        help="Image size for training (default: 640)")
    parser.add_argument("--device", default="", 
                        help="Device to use for training (default: auto-select)")
    parser.add_argument("--webcam", action="store_true",
                        help="Run inference on webcam feed")
    parser.add_argument("--model_path", default="best.pt",
                        help="Path to the model for webcam inference (default: best.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for webcam inference (default: 0.25)")
    return parser.parse_args()


def train_model(
    data_yaml: str,
    output_dir: str,
    model_name: str,
    base_model: str = "yolov8n.pt",
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


def run_webcam_inference(model_path: str, conf_threshold: float = 0.25, device: str = ""):
    """
    Run real-time inference on webcam feed using a trained YOLOv8 model.
    
    Args:
        model_path: Path to the trained model weights
        conf_threshold: Confidence threshold for detections
        device: Device to use for inference
    """
    print(f"\n===== Running Webcam Inference with Model: {model_path} =====\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set webcam properties for better quality (if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam inference started. Press 'q' to quit.")
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                break
            
            # Run inference on the frame
            results = model.predict(frame, conf=conf_threshold, device=device)
            
            # Visualize results on the frame
            annotated_frame = results[0].plot()
            
            # Display frame with annotations
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam inference stopped.")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # If webcam mode is selected, run inference on webcam
    if args.webcam:
        run_webcam_inference(args.model_path, args.conf, args.device)
        return
    
    # Ensure models_dir is provided for training mode
    if not args.models_dir and not args.webcam:
        print("Error: --models_dir is required for training mode.")
        return
    
    # Create models output directory
    models_dir = Path(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    training_info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
    }
    
    # Train door detection model if door_data is provided
    if args.door_data:
        print("\n===== Training Door Detection Model =====\n")
        door_model_path = train_model(
            args.door_data,
            str(models_dir),
            "door_detection",
            args.base_model,
            args.epochs,
            args.batch_size,
            args.img_size,
            args.device
        )
        training_info["door_model"] = door_model_path
        training_info["door_data"] = args.door_data
    
    # Train uniformat models if data is provided
    if args.level3_data:
        print("\n===== Training Uniformat Level 3 Model =====\n")
        level3_model_path = train_model(
            args.level3_data,
            str(models_dir),
            "uniformat_level3",
            args.base_model,
            args.epochs,
            args.batch_size,
            args.img_size,
            args.device
        )
        training_info["level3_model"] = level3_model_path
        training_info["level3_data"] = args.level3_data
        
    if args.level4_data:
        print("\n===== Training Uniformat Level 4 Model =====\n")
        level4_model_path = train_model(
            args.level4_data,
            str(models_dir),
            "uniformat_level4",
            args.base_model,
            args.epochs,
            args.batch_size,
            args.img_size,
            args.device
        )
        training_info["level4_model"] = level4_model_path
        training_info["level4_data"] = args.level4_data
    
    # Validate that at least one dataset was provided
    if not any([args.level3_data, args.level4_data, args.door_data]):
        print("Error: You must provide at least one dataset to train.")
        print("Use --level3_data, --level4_data, or --door_data")
        return
    
    # Save training info
    with open(models_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n===== Training Complete =====")
    print(f"Training info saved to: {models_dir / 'training_info.json'}")
    print("\nYou can now use these models with the inference script or run with --webcam option.")


if __name__ == "__main__":
    main()