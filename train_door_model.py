#!/usr/bin/env python3
# filepath: train_door_model.py

"""
Training Script for Door Detection Model

This script fine-tunes a YOLOv8 model for door detection using the doorDataset folder.

Usage:
    python train_door_model.py [--epochs 10] [--batch_size 16] [--img_size 640]
                              [--device ""] [--models_dir ./models]

 # For webcam inference:
    python train_models.py --webcam --model_path best.pt
    
    # For iPhone camera feed inference:
    python train_models.py --webcam --camera_url "http://192.168.1.172:8888/video" --model_path best2.pt 
                              
                              """

import argparse
import os
import json
import time
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


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Door dataset path (use absolute path)
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
    print("\nYou can now use this model with the inference script or run with --webcam option.")


if __name__ == "__main__":
    main()