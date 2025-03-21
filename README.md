# Uniformat Codes Horizon - Building Element Classification

A computer vision system that automatically detects and classifies building elements according to Uniformat classification codes at both level 3 and level 4 in videos and images.

## Overview

This project uses YOLOv8 to detect and classify building elements in architectural videos and images. The system is trained to recognize elements according to the Uniformat classification system at two levels of detail:

- **Uniformat Level 3:** A coarser grouping capturing critical building element distinctions (e.g., B2030 - Exterior Doors)
- **Uniformat Level 4:** A more detailed classification that preserves finer granularity (e.g., B2030.10 - Exterior Entrance Doors)

## Features

- Processes video files to detect building elements
- Classifies elements according to Uniformat level 3 and level 4 codes
- Visualizes detections with color-coded bounding boxes and labels
- Supports displaying either level 3, level 4, or both classification levels
- Includes tools for preprocessing training data and fine-tuning YOLOv8 models

## Project Structure

- `app.py` - Main video processing script
- `preprocess_data.py` - Data preprocessing script for preparing training datasets
- `train_models.py` - Training script for fine-tuning YOLOv8 models
- `yolov8n.pt` - Base YOLOv8 nano model

## Requirements

```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
tqdm>=4.65.0
pyyaml>=6.0.0
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/UniformatCodesHorizon.git
   cd UniformatCodesHorizon
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Preprocessing Data for Training

This step prepares your datasets (FloorPlanCAD and ZInD) for training:

```bash
python preprocess_data.py --floorplancad_path /path/to/floorplancad 
                         --zind_path /path/to/zind 
                         --output_path /path/to/output
                         --val_split 0.2
                         --test_split 0.1
```

Arguments:
- `--floorplancad_path`: Path to FloorPlanCAD dataset
- `--zind_path`: Path to ZInD dataset
- `--output_path`: Path to save processed datasets
- `--val_split`: Validation split ratio (default: 0.2)
- `--test_split`: Test split ratio (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)

### Training Models

Once data is preprocessed, you can train the YOLOv8 models:

```bash
python train_models.py --level3_data /path/to/level3/data.yaml 
                      --level4_data /path/to/level4/data.yaml
                      --models_dir /path/to/save/models
                      --base_model yolov8n.pt
                      --epochs 100
```

Arguments:
- `--level3_data`: Path to level 3 data.yaml file
- `--level4_data`: Path to level 4 data.yaml file
- `--models_dir`: Directory to save trained models
- `--base_model`: Base YOLOv8 model to fine-tune (default: yolov8n.pt)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--img_size`: Image size for training (default: 640)
- `--device`: Device to use for training (default: auto-select)

### Running Inference

Process a video using the trained models:

```bash
python app.py -i input.mp4 -o output_video.mp4 
             -m3 models/uniformat_level3/weights/best.pt 
             -m4 models/uniformat_level4/weights/best.pt 
             --mapping_file /path/to/output/uniformat_mapping.json
             --level both
             --display
```

Arguments:
- `-i, --input`: Path to input video file
- `-o, --output`: Path to output video file
- `-m3, --level3_model`: Path to YOLOv8 model for level 3 classification
- `-m4, --level4_model`: Path to YOLOv8 model for level 4 classification
- `--mapping_file`: Path to Uniformat mapping JSON file (generated during preprocessing)
- `-c, --confidence`: Confidence threshold for detections (default: 0.5)
- `--level`: Which Uniformat level to display ("3", "4", or "both", default: "both")
- `--display`: Display video during processing

## Uniformat Classification

The Uniformat system is a method of arranging construction information based on functional elements. This project currently supports the following major categories:

- **B1010** - Floor Construction
- **B1020** - Roof Construction
- **B2010** - Exterior Walls
- **B2020** - Exterior Windows
- **B2030** - Exterior Doors
- **B3010** - Roof Coverings
- **B3020** - Roof Openings
- **C1010** - Partitions
- **C1020** - Interior Doors
- **C1030** - Fittings
- **C2010** - Stair Construction
- **C3030** - Ceiling Finishes

Each level 3 code (e.g., B2030) is further broken down into level 4 codes (e.g., B2030.10, B2030.20) for more detailed classification.

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework
- FloorPlanCAD and ZInD datasets are used for training the models
