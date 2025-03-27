import os
import json
import shutil
from pathlib import Path
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
base_dir = Path('/Users/siddhantbadola/UniformatCodesHorizon/uniPyV1')
dataset_dir = base_dir / 'DatasetC10' / 'C102022 Door'
data2_annotations = dataset_dir / 'data (2)' / 'annotations' / 'instances_Train.json'
data3_annotations = dataset_dir / 'data (3)' / 'annotations' / 'instances_default.json'
data2_images = dataset_dir / 'data (2)' / 'images' / 'Train'
data3_images = dataset_dir / 'data (3)' / 'images' / 'default'

# Define output directories
output_dir = base_dir / 'merged_dataset'
output_train_dir = output_dir / 'images' / 'train'
output_val_dir = output_dir / 'images' / 'val'
output_test_dir = output_dir / 'images' / 'test'
output_annotations_dir = output_dir / 'annotations'

# Create output directories if they don't exist
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_val_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

# Split ratio
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def merge_datasets():
    logging.info("Starting dataset merge process")
    
    # Verify paths exist
    for path in [data2_annotations, data3_annotations, data2_images, data3_images]:
        if not path.exists():
            logging.error(f"Path does not exist: {path}")
            return False
    
    # Load annotation files
    try:
        logging.info(f"Loading annotations from {data2_annotations}")
        data2_anno = load_json(data2_annotations)
        
        logging.info(f"Loading annotations from {data3_annotations}")
        data3_anno = load_json(data3_annotations)
    except Exception as e:
        logging.error(f"Failed to load annotation files: {e}")
        return False
    
    # Check what files we actually have
    logging.info(f"Dataset 2 contains {len(data2_anno.get('images', []))} images")
    logging.info(f"Dataset 3 contains {len(data3_anno.get('images', []))} images")
    
    # Initialize merged annotation structure
    merged_anno = {
        "licenses": data2_anno.get("licenses", []),
        "info": data2_anno.get("info", {}),
        "categories": data2_anno.get("categories", []),
        "images": [],
        "annotations": []
    }
    
    # Combine categories (avoiding duplicates)
    category_ids = set(cat["id"] for cat in merged_anno["categories"])
    for cat in data3_anno.get("categories", []):
        if cat["id"] not in category_ids:
            merged_anno["categories"].append(cat)
            category_ids.add(cat["id"])
    
    # Prepare to track image IDs as we merge
    next_image_id = 1
    image_id_map = {}  # Maps original image IDs to new IDs
    
    # Build a list of all images to be randomly split later
    all_images = []
    
    # Process images from data2
    logging.info(f"Processing images from {data2_images}")
    for img_info in data2_anno.get("images", []):
        old_id = img_info["id"]
        filename = img_info["file_name"]
        src_path = data2_images / filename
        
        if not src_path.exists():
            logging.warning(f"Image file not found: {src_path}")
            continue
            
        img_info["id"] = next_image_id
        image_id_map[(1, old_id)] = next_image_id  # Use tuple to distinguish sources
        next_image_id += 1
        
        all_images.append((img_info, src_path, 1, old_id))  # Store info for later processing
    
    # Process images from data3
    logging.info(f"Processing images from {data3_images}")
    for img_info in data3_anno.get("images", []):
        old_id = img_info["id"]
        filename = img_info["file_name"]
        src_path = data3_images / filename
        
        if not src_path.exists():
            logging.warning(f"Image file not found: {src_path}")
            continue
            
        img_info["id"] = next_image_id
        image_id_map[(2, old_id)] = next_image_id  # Use tuple to distinguish sources
        next_image_id += 1
        
        all_images.append((img_info, src_path, 2, old_id))  # Store info for later processing
    
    # Shuffle and split images into train/val/test
    random.shuffle(all_images)
    total_images = len(all_images)
    train_count = int(total_images * TRAIN_RATIO)
    val_count = int(total_images * VAL_RATIO)
    
    train_images = all_images[:train_count]
    val_images = all_images[train_count:train_count+val_count]
    test_images = all_images[train_count+val_count:]
    
    # Process and copy all splits
    train_anno = {"licenses": merged_anno["licenses"], "info": merged_anno["info"], 
                  "categories": merged_anno["categories"], "images": [], "annotations": []}
    val_anno = {"licenses": merged_anno["licenses"], "info": merged_anno["info"], 
                "categories": merged_anno["categories"], "images": [], "annotations": []}
    test_anno = {"licenses": merged_anno["licenses"], "info": merged_anno["info"], 
                 "categories": merged_anno["categories"], "images": [], "annotations": []}
    
    # Process train images
    logging.info(f"Processing {len(train_images)} train images")
    for img_info, src_path, dataset_id, old_id in train_images:
        dst_path = output_train_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
            merged_anno["images"].append(img_info)
            train_anno["images"].append(img_info)
        except Exception as e:
            logging.error(f"Failed to copy {src_path}: {e}")
            continue
    
    # Process validation images
    logging.info(f"Processing {len(val_images)} validation images")
    for img_info, src_path, dataset_id, old_id in val_images:
        dst_path = output_val_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
            merged_anno["images"].append(img_info)
            val_anno["images"].append(img_info)
        except Exception as e:
            logging.error(f"Failed to copy {src_path}: {e}")
            continue
    
    # Process test images
    logging.info(f"Processing {len(test_images)} test images")
    for img_info, src_path, dataset_id, old_id in test_images:
        dst_path = output_test_dir / src_path.name
        try:
            shutil.copy2(src_path, dst_path)
            merged_anno["images"].append(img_info)
            test_anno["images"].append(img_info)
        except Exception as e:
            logging.error(f"Failed to copy {src_path}: {e}")
            continue
    
    # Process annotations from data2
    next_anno_id = 1
    logging.info("Processing annotations from dataset 2")
    for anno in data2_anno.get("annotations", []):
        source_img_id = (1, anno["image_id"])
        if source_img_id in image_id_map:
            anno["id"] = next_anno_id
            anno["image_id"] = image_id_map[source_img_id]
            merged_anno["annotations"].append(anno)
            
            # Add to the appropriate split
            for split_anno in [train_anno, val_anno, test_anno]:
                if any(img["id"] == anno["image_id"] for img in split_anno["images"]):
                    split_anno["annotations"].append(anno)
                    break
                    
            next_anno_id += 1
    
    # Process annotations from data3
    logging.info("Processing annotations from dataset 3")
    for anno in data3_anno.get("annotations", []):
        source_img_id = (2, anno["image_id"])
        if source_img_id in image_id_map:
            anno["id"] = next_anno_id
            anno["image_id"] = image_id_map[source_img_id]
            merged_anno["annotations"].append(anno)
            
            # Add to the appropriate split
            for split_anno in [train_anno, val_anno, test_anno]:
                if any(img["id"] == anno["image_id"] for img in split_anno["images"]):
                    split_anno["annotations"].append(anno)
                    break
                    
            next_anno_id += 1
    
    # Save merged annotations
    logging.info("Saving annotations")
    save_json(merged_anno, output_annotations_dir / "instances_merged.json")
    save_json(train_anno, output_annotations_dir / "instances_train.json")
    save_json(val_anno, output_annotations_dir / "instances_val.json")
    save_json(test_anno, output_annotations_dir / "instances_test.json")
    
    # Create data.yaml for YOLOv5 training
    categories = [cat["name"] for cat in merged_anno["categories"]]
    yaml_content = {
        "train": "./images/train",
        "val": "./images/val",
        "test": "./images/test",
        "nc": len(categories),
        "names": categories
    }
    
    with open(output_dir / "data.yaml", "w") as f:
        f.write("# Dataset merged from C102022 Door datasets\n")
        for key, value in yaml_content.items():
            if key == "names":
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    logging.info(f"Dataset merge complete. Merged {len(merged_anno['images'])} images and {len(merged_anno['annotations'])} annotations.")
    logging.info(f"Train: {len(train_anno['images'])} images, {len(train_anno['annotations'])} annotations")
    logging.info(f"Validation: {len(val_anno['images'])} images, {len(val_anno['annotations'])} annotations")
    logging.info(f"Test: {len(test_anno['images'])} images, {len(test_anno['annotations'])} annotations")
    logging.info(f"Output directory: {output_dir}")
    
    return True

if __name__ == "__main__":
    logging.info("Script started")
    try:
        success = merge_datasets()
        if not success:
            logging.error("Dataset merge failed")
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    logging.info("Script completed")