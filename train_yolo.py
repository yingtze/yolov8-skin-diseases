#!/usr/bin/env python3
"""
YOLOv8 Training Script for Skin Diseases Classification
Trains YOLOv8 small model on the prepared dataset
"""

from ultralytics import YOLO
import torch
from pathlib import Path

def main():
    """Main training function"""
    print("="*60)
    print("YOLOv8 Small - Skin Diseases Classification Training")
    print("="*60)
    
    # Check for GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load YOLOv8 small classification model
    print("\nLoading YOLOv8 small classification model...")
    model = YOLO('yolov8s-cls.pt')  # YOLOv8 small classification
    
    # Training configuration
    print("\nTraining Configuration:")
    config = {
        'data': 'dataset',  # Path to dataset directory
        'epochs': 50,
        'imgsz': 224,
        'batch': 16,
        'device': device,
        'project': 'runs/classify',
        'name': 'skin_diseases',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'patience': 10,  # Early stopping patience
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'verbose': True,
        'plots': True,
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Start training
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60 + "\n")
    
    results = model.train(**config)
    
    # Training complete
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Export model
    print("\nExporting model...")
    model_path = Path('runs/classify/skin_diseases/weights/best.pt')
    
    if model_path.exists():
        best_model = YOLO(model_path)
        
        # Export to ONNX format
        print("  Exporting to ONNX format...")
        best_model.export(format='onnx')
        
        print("\n✓ Model exported successfully!")
        print(f"  Best model: {model_path}")
        print(f"  ONNX model: {model_path.parent / 'best.onnx'}")
    
    print("\nNext step: Run test_model.py to evaluate the model")

if __name__ == "__main__":
    main()
