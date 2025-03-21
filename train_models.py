#!/usr/bin/env python3
# filepath: train_models.py

"""
Training Script for Uniformat Classification Models

This script fine-tunes YOLOv8 models for both Uniformat level 3 and level 4 building element
classification using the datasets prepared by the preprocess_data.py script.

Usage:
    python train_models.py --level3_data /path/to/level3/data.yaml 
                          --level4_data /path/to/level4/data.yaml
                          --models_dir /path/to/save/models
"""

import argparse
import os
import json
import time
from pathlib import Path

from ultralytics import YOLO

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Uniformat Classification Model Trainer")
    parser.add_argument("--level3_data", required=True, 
                        help="Path to level 3 data.yaml file")
    parser.add_argument("--level4_data", required=True, 
                        help="Path to level 4 data.yaml file")
    parser.add_argument("--models_dir", required=True, 
                        help="Directory to save trained models")
    parser.add_argument("--base_model", default="yolov8n.pt", 
                        help="Base YOLOv8 model to fine-tune (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of training epochs (default: 100)")
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
    base_model: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = ""
) -> str:
    """
    Train a YOLOv8 model.
    
    Args:
        data_yaml: Path to the data.yaml file
        output_dir: Directory to save the trained model
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
        name=f"uniformat_{Path(data_yaml).parent.name}",
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    # Get the path to the best model weights
    best_weights_path = Path(output_dir) / f"uniformat_{Path(data_yaml).parent.name}" / "weights" / "best.pt"
    
    print(f"Training complete. Best weights saved to: {best_weights_path}")
    
    return str(best_weights_path)


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Create models output directory
    models_dir = Path(args.models_dir)
    os.makedirs(models_dir, exist_ok=True)
    
    # Train level 3 model
    print("\n===== Training Uniformat Level 3 Model =====\n")
    level3_model_path = train_model(
        args.level3_data,
        str(models_dir),
        args.base_model,
        args.epochs,
        args.batch_size,
        args.img_size,
        args.device
    )
    
    # Train level 4 model
    print("\n===== Training Uniformat Level 4 Model =====\n")
    level4_model_path = train_model(
        args.level4_data,
        str(models_dir),
        args.base_model,
        args.epochs,
        args.batch_size,
        args.img_size,
        args.device
    )
    
    # Create a simple metadata file with training info
    training_info = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "level3_model": level3_model_path,
        "level4_model": level4_model_path,
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "level3_data": args.level3_data,
        "level4_data": args.level4_data
    }
    
    # Save training info
    with open(models_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\n===== Training Complete =====")
    print(f"Level 3 model saved to: {level3_model_path}")
    print(f"Level 4 model saved to: {level4_model_path}")
    print(f"Training info saved to: {models_dir / 'training_info.json'}")
    print("\nYou can now use these models with the inference script.")


if __name__ == "__main__":
    main()