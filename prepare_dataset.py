#!/usr/bin/env python3
"""
Dataset Preparation Script for YOLO Classification
Prepares skin diseases dataset for YOLOv8 training
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Set random seed for reproducibility
random.seed(42)

# Configuration
SOURCE_DIR = "train"
OUTPUT_DIR = "dataset"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# Class names
CLASSES = ["acne", "eksim", "herpes", "panu", "rosacea"]

def create_directory_structure():
    """Create YOLO classification directory structure"""
    print("Creating directory structure...")
    
    for split in ["train", "val", "test"]:
        for class_name in CLASSES:
            path = Path(OUTPUT_DIR) / split / class_name
            path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure in {OUTPUT_DIR}/")

def split_and_copy_images():
    """Split images into train/val/test and copy to respective directories"""
    print("\nSplitting and copying images...")
    
    stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    
    for class_name in CLASSES:
        source_path = Path(SOURCE_DIR) / class_name
        
        # Get all image files
        images = list(source_path.glob("*.jpg")) + \
                 list(source_path.glob("*.jpeg")) + \
                 list(source_path.glob("*.png"))
        
        # Shuffle images
        random.shuffle(images)
        
        total = len(images)
        train_count = int(total * TRAIN_RATIO)
        val_count = int(total * VAL_RATIO)
        
        # Split indices
        train_images = images[:train_count]
        val_images = images[train_count:train_count + val_count]
        test_images = images[train_count + val_count:]
        
        # Copy images to respective directories
        for img in train_images:
            dest = Path(OUTPUT_DIR) / "train" / class_name / img.name
            shutil.copy2(img, dest)
            stats[class_name]["train"] += 1
        
        for img in val_images:
            dest = Path(OUTPUT_DIR) / "val" / class_name / img.name
            shutil.copy2(img, dest)
            stats[class_name]["val"] += 1
        
        for img in test_images:
            dest = Path(OUTPUT_DIR) / "test" / class_name / img.name
            shutil.copy2(img, dest)
            stats[class_name]["test"] += 1
        
        print(f"  {class_name}: {total} images → "
              f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    
    return stats

def print_statistics(stats):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total_train = sum(s["train"] for s in stats.values())
    total_val = sum(s["val"] for s in stats.values())
    total_test = sum(s["test"] for s in stats.values())
    total = total_train + total_val + total_test
    
    print(f"\nTotal images: {total}")
    print(f"  Train: {total_train} ({total_train/total*100:.1f}%)")
    print(f"  Val:   {total_val} ({total_val/total*100:.1f}%)")
    print(f"  Test:  {total_test} ({total_test/total*100:.1f}%)")
    
    print(f"\nPer-class distribution:")
    print(f"{'Class':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 60)
    
    for class_name in CLASSES:
        s = stats[class_name]
        class_total = s["train"] + s["val"] + s["test"]
        print(f"{class_name:<12} {s['train']:<8} {s['val']:<8} {s['test']:<8} {class_total:<8}")
    
    print("="*60)

def main():
    """Main execution function"""
    print("="*60)
    print("YOLO Dataset Preparation")
    print("="*60)
    
    # Check if source directory exists
    if not Path(SOURCE_DIR).exists():
        print(f"Error: Source directory '{SOURCE_DIR}' not found!")
        return
    
    # Create directory structure
    create_directory_structure()
    
    # Split and copy images
    stats = split_and_copy_images()
    
    # Print statistics
    print_statistics(stats)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"✓ Output directory: {OUTPUT_DIR}/")
    print(f"\nNext step: Create data.yaml configuration file")

if __name__ == "__main__":
    main()
