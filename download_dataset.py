#!/usr/bin/env python3
"""
Dataset Download Script using KaggleHub
Downloads and sets up the skin diseases dataset for YOLO training
"""

import os
import shutil
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Error: kagglehub not installed.")
    print("Please install it with: pip install kagglehub")
    exit(1)

# Configuration
DATASET_NAME = "sponishflea/classification-of-skin-diseases"
TARGET_DIR = "train"
EXPECTED_CLASSES = ["acne", "eksim", "herpes", "panu", "rosacea"]

def download_dataset():
    """Download dataset using kagglehub"""
    print("="*60)
    print("Downloading Skin Diseases Dataset")
    print("="*60)
    print(f"\nDataset: {DATASET_NAME}")
    print("This may take a few minutes...\n")
    
    try:
        # Download dataset - returns path to downloaded files
        path = kagglehub.dataset_download(DATASET_NAME)
        print(f"\n✓ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged into Kaggle")
        print("2. Accept the dataset terms at: https://www.kaggle.com/datasets/sponishflea/classification-of-skin-diseases")
        print("3. Check your internet connection")
        return None

def setup_train_directory(source_path):
    """Set up train/ directory with proper structure"""
    print("\n" + "="*60)
    print("Setting Up Train Directory")
    print("="*60)
    
    source_path = Path(source_path)
    target_path = Path(TARGET_DIR)
    
    # Check if train directory already exists in downloaded data
    possible_train_paths = [
        source_path / "train",
        source_path,
    ]
    
    train_source = None
    for path in possible_train_paths:
        if path.exists():
            # Check if it has the expected class subdirectories
            subdirs = [d.name for d in path.iterdir() if d.is_dir()]
            if any(cls in subdirs for cls in EXPECTED_CLASSES):
                train_source = path
                break
    
    if train_source is None:
        print(f"\n✗ Could not find train directory with expected classes")
        print(f"Expected classes: {EXPECTED_CLASSES}")
        print(f"Searched in: {source_path}")
        return False
    
    print(f"\nFound data at: {train_source}")
    
    # Create or clean target directory
    if target_path.exists():
        print(f"\n⚠ Target directory '{TARGET_DIR}/' already exists")
        response = input("Do you want to overwrite it? (y/n): ").lower()
        if response != 'y':
            print("Aborted.")
            return False
        shutil.rmtree(target_path)
        print(f"✓ Removed existing '{TARGET_DIR}/' directory")
    
    # Copy or move the train directory
    print(f"\nCopying data to '{TARGET_DIR}/'...")
    
    if train_source == source_path:
        # Source is the root, need to copy class directories
        target_path.mkdir(exist_ok=True)
        for class_name in EXPECTED_CLASSES:
            src_class = train_source / class_name
            if src_class.exists():
                dst_class = target_path / class_name
                shutil.copytree(src_class, dst_class)
                count = len(list(dst_class.glob("*.*")))
                print(f"  ✓ {class_name}: {count} images")
            else:
                print(f"  ⚠ {class_name}: not found")
    else:
        # Source has a train directory, copy it
        shutil.copytree(train_source, target_path)
        print(f"✓ Copied '{train_source}' to '{TARGET_DIR}/'")
    
    return True

def verify_structure():
    """Verify the train directory structure"""
    print("\n" + "="*60)
    print("Verifying Directory Structure")
    print("="*60)
    
    train_path = Path(TARGET_DIR)
    
    if not train_path.exists():
        print(f"\n✗ '{TARGET_DIR}/' directory not found")
        return False
    
    print(f"\nChecking '{TARGET_DIR}/' directory...")
    
    all_found = True
    total_images = 0
    
    for class_name in EXPECTED_CLASSES:
        class_path = train_path / class_name
        if class_path.exists() and class_path.is_dir():
            # Count images
            images = list(class_path.glob("*.jpg")) + \
                    list(class_path.glob("*.jpeg")) + \
                    list(class_path.glob("*.png"))
            count = len(images)
            total_images += count
            print(f"  ✓ {class_name}: {count} images")
        else:
            print(f"  ✗ {class_name}: NOT FOUND")
            all_found = False
    
    print(f"\nTotal images: {total_images}")
    
    if all_found and total_images > 0:
        print("\n✓ Directory structure verified successfully!")
        return True
    else:
        print("\n✗ Directory structure verification failed")
        return False

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("Skin Diseases Dataset Downloader")
    print("="*60)
    
    # Step 1: Download dataset
    download_path = download_dataset()
    if download_path is None:
        print("\n✗ Download failed. Exiting.")
        return
    
    # Step 2: Set up train directory
    success = setup_train_directory(download_path)
    if not success:
        print("\n✗ Setup failed. Exiting.")
        return
    
    # Step 3: Verify structure
    verified = verify_structure()
    
    if verified:
        print("\n" + "="*60)
        print("Setup Complete!")
        print("="*60)
        print(f"\n✓ Dataset ready in '{TARGET_DIR}/' directory")
        print("\nNext steps:")
        print("  1. Run: python3 prepare_dataset.py")
        print("  2. Run: python3 train_yolo.py")
        print("  3. Run: python3 test_model.py")
    else:
        print("\n✗ Setup incomplete. Please check the errors above.")

if __name__ == "__main__":
    main()
