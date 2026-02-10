#!/usr/bin/env python3
"""
Inference Showcase Script
Runs predictions on sample images and creates a showcase markdown file
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import random

# Configuration
MODEL_PATH = 'runs/classify/runs/classify/skin_diseases/weights/best.pt'
TEST_DIR = Path('dataset/test')
SHOWCASE_DIR = Path('showcase')
SAMPLES_PER_CLASS = 3
CLASS_NAMES = ["acne", "eksim", "herpes", "panu", "rosacea"]

def setup_showcase_directory():
    """Create showcase directory"""
    SHOWCASE_DIR.mkdir(exist_ok=True)
    images_dir = SHOWCASE_DIR / 'images'
    images_dir.mkdir(exist_ok=True)
    return images_dir

def load_model():
    """Load the trained model"""
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    return model

def get_sample_images():
    """Get sample images from each class"""
    samples = {}
    
    for class_name in CLASS_NAMES:
        class_dir = TEST_DIR / class_name
        
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found")
            continue
        
        # Get all images
        images = list(class_dir.glob('*.jpg')) + \
                 list(class_dir.glob('*.jpeg')) + \
                 list(class_dir.glob('*.png'))
        
        # Randomly sample
        if len(images) > SAMPLES_PER_CLASS:
            samples[class_name] = random.sample(images, SAMPLES_PER_CLASS)
        else:
            samples[class_name] = images
    
    return samples

def run_inference(model, samples, images_dir):
    """Run inference on sample images"""
    results = []
    
    for class_name, image_paths in samples.items():
        print(f"\nProcessing {class_name}...")
        
        for img_path in image_paths:
            # Run prediction
            pred_results = model(img_path, verbose=False)
            
            # Get prediction details
            pred_class_idx = pred_results[0].probs.top1
            pred_class_name = CLASS_NAMES[pred_class_idx]
            confidence = pred_results[0].probs.top1conf.item()
            
            # Get top 3 predictions
            top3_indices = pred_results[0].probs.top5[:3]
            top3_probs = pred_results[0].probs.top5conf[:3]
            
            # Copy image to showcase directory
            dest_path = images_dir / f"{class_name}_{img_path.name}"
            shutil.copy2(img_path, dest_path)
            
            # Store result
            result = {
                'true_class': class_name,
                'pred_class': pred_class_name,
                'confidence': confidence,
                'correct': class_name == pred_class_name,
                'image_path': dest_path,
                'top3': [(CLASS_NAMES[idx], prob.item()) for idx, prob in zip(top3_indices, top3_probs)]
            }
            results.append(result)
            
            status = "✓" if result['correct'] else "✗"
            print(f"  {status} {img_path.name}: {pred_class_name} ({confidence:.2%})")
    
    return results

def create_showcase_markdown(results):
    """Create showcase markdown file"""
    md_content = []
    
    # Header
    md_content.append("# YOLOv8 Skin Diseases Classification - Inference Showcase\n")
    md_content.append("Real predictions from the trained YOLOv8 small model achieving **99.34% accuracy**.\n")
    
    # Statistics
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    md_content.append(f"## Showcase Statistics\n")
    md_content.append(f"- **Total Predictions**: {total}")
    md_content.append(f"- **Correct**: {correct}")
    md_content.append(f"- **Accuracy**: {accuracy:.2f}%\n")
    
    # Group by class
    md_content.append("## Predictions by Class\n")
    
    for class_name in CLASS_NAMES:
        class_results = [r for r in results if r['true_class'] == class_name]
        
        if not class_results:
            continue
        
        md_content.append(f"### {class_name.capitalize()}\n")
        
        for result in class_results:
            img_rel_path = f"showcase/images/{result['image_path'].name}"
            
            # Image
            md_content.append(f"![{result['image_path'].name}]({img_rel_path})\n")
            
            # Prediction details
            status_emoji = "✅" if result['correct'] else "❌"
            md_content.append(f"**{status_emoji} Prediction**: {result['pred_class']} ({result['confidence']:.2%} confidence)\n")
            
            # Top 3 predictions
            md_content.append("**Top 3 Predictions**:")
            for i, (cls, prob) in enumerate(result['top3'], 1):
                md_content.append(f"  {i}. {cls}: {prob:.2%}")
            
            md_content.append("")  # Empty line
    
    # Model info
    md_content.append("---\n")
    md_content.append("## Model Information\n")
    md_content.append("- **Architecture**: YOLOv8 Small Classification")
    md_content.append("- **Training Dataset**: 1,044 images across 5 classes")
    md_content.append("- **Test Accuracy**: 99.34%")
    md_content.append("- **Model Size**: 9.8MB")
    md_content.append("- **Classes**: acne, eksim, herpes, panu, rosacea\n")
    
    # Write to file
    output_path = SHOWCASE_DIR / 'INFERENCE_SHOWCASE.md'
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_content))
    
    print(f"\n✓ Showcase markdown created: {output_path}")
    return output_path

def main():
    """Main execution"""
    print("="*60)
    print("YOLOv8 Inference Showcase Generator")
    print("="*60)
    
    # Setup
    images_dir = setup_showcase_directory()
    
    # Load model
    model = load_model()
    
    # Get sample images
    print("\nSelecting sample images...")
    samples = get_sample_images()
    total_samples = sum(len(imgs) for imgs in samples.values())
    print(f"Selected {total_samples} sample images")
    
    # Run inference
    print("\nRunning inference...")
    results = run_inference(model, samples, images_dir)
    
    # Create markdown
    print("\nCreating showcase markdown...")
    output_path = create_showcase_markdown(results)
    
    print("\n" + "="*60)
    print("Showcase Generation Complete!")
    print("="*60)
    print(f"\nFiles created:")
    print(f"  - {output_path}")
    print(f"  - {len(results)} images in showcase/images/")
    print(f"\nYou can now upload the showcase/ directory to Kaggle!")

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    main()
