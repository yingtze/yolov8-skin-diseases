#!/usr/bin/env python3
"""
YOLOv8 Model Testing and Evaluation Script
Tests the trained model on the test set and generates metrics
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Class names
CLASSES = ["acne", "eksim", "herpes", "panu", "rosacea"]

def load_model():
    """Load the best trained model"""
    model_path = Path('runs/classify/runs/classify/skin_diseases/weights/best.pt')
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train_yolo.py")
        return None
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    return model

def evaluate_on_test_set(model):
    """Evaluate model on test set"""
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60 + "\n")
    
    # Validate on test set
    results = model.val(data='dataset', split='test')
    
    return results

def predict_and_analyze(model):
    """Run predictions on test set and create confusion matrix"""
    print("\nRunning predictions on test set...")
    
    test_dir = Path('dataset/test')
    y_true = []
    y_pred = []
    
    # Iterate through each class
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = test_dir / class_name
        
        if not class_dir.exists():
            continue
        
        # Get all images in this class
        images = list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.jpeg')) + \
                 list(class_dir.glob('*.png'))
        
        for img_path in images:
            # Predict
            results = model(img_path, verbose=False)
            
            # Get predicted class
            pred_class = results[0].probs.top1
            
            y_true.append(class_idx)
            y_pred.append(pred_class)
    
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred):
    """Plot and save confusion matrix"""
    print("\nGenerating confusion matrix...")
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Skin Diseases Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    output_path = 'runs/classify/runs/classify/skin_diseases/confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to {output_path}")
    
    plt.close()

def print_classification_report(y_true, y_pred):
    """Print detailed classification report"""
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60 + "\n")
    
    report = classification_report(y_true, y_pred, 
                                   target_names=CLASSES,
                                   digits=4)
    print(report)
    
    # Save report to file
    output_path = 'runs/classify/runs/classify/skin_diseases/classification_report.txt'
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Classification report saved to {output_path}")

def test_sample_predictions(model, num_samples=5):
    """Test and visualize sample predictions"""
    print(f"\nTesting {num_samples} sample predictions per class...")
    
    test_dir = Path('dataset/test')
    
    for class_name in CLASSES:
        class_dir = test_dir / class_name
        
        if not class_dir.exists():
            continue
        
        # Get sample images
        images = list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.jpeg')) + \
                 list(class_dir.glob('*.png'))
        
        samples = images[:min(num_samples, len(images))]
        
        print(f"\n{class_name}:")
        for img_path in samples:
            results = model(img_path, verbose=False)
            pred_class = results[0].probs.top1
            confidence = results[0].probs.top1conf.item()
            
            pred_name = CLASSES[pred_class]
            status = "✓" if pred_name == class_name else "✗"
            print(f"  {status} {img_path.name}: {pred_name} ({confidence:.2%})")

def main():
    """Main testing function"""
    print("="*60)
    print("YOLOv8 Model Testing and Evaluation")
    print("="*60)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Evaluate on test set
    results = evaluate_on_test_set(model)
    
    # Run predictions and get true/pred labels
    y_true, y_pred = predict_and_analyze(model)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    # Test sample predictions
    test_sample_predictions(model)
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    print(f"\nResults saved in: runs/classify/runs/classify/skin_diseases/")

if __name__ == "__main__":
    main()
