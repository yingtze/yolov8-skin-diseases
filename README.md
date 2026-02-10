# Skin Diseases Classification with YOLOv8

A YOLOv8 small model trained to classify 5 types of skin diseases with **99.34% accuracy**.

## Dataset

- **Source**: [Kaggle - Classification of Skin Diseases](https://www.kaggle.com/datasets/sponishflea/classification-of-skin-diseases)
- **Total Images**: 1,494
- **Classes**: acne, eksim, herpes, panu, rosacea
- **Split**: 70% train / 20% val / 10% test

## Model Performance

- **Accuracy**: 99.34%
- **Model Size**: 9.8MB
- **Architecture**: YOLOv8 small classification

### Per-Class Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| acne | 100.00% | 100.00% | 100.00% |
| eksim | 100.00% | 96.77% | 98.36% |
| herpes | 96.77% | 100.00% | 98.36% |
| panu | 100.00% | 100.00% | 100.00% |
| rosacea | 100.00% | 100.00% | 100.00% |

## Quick Start

### 1. Prepare Dataset
```bash
python3 prepare_dataset.py
```

### 2. Train Model
```bash
python3 train_yolo.py
```

### 3. Evaluate Model
```bash
python3 test_model.py
```

## Usage

### Inference on New Images
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/classify/runs/classify/skin_diseases/weights/best.pt')

# Predict
results = model('path/to/image.jpg')
pred_class = results[0].probs.top1
confidence = results[0].probs.top1conf.item()

class_names = ["acne", "eksim", "herpes", "panu", "rosacea"]
print(f"Predicted: {class_names[pred_class]} ({confidence:.2%})")
```

## Requirements

```bash
pip install ultralytics kaggle opencv-python pillow matplotlib scikit-learn seaborn
```

## Project Structure

```
trainML/
├── prepare_dataset.py      # Dataset preparation script
├── train_yolo.py           # Training script
├── test_model.py           # Evaluation script
├── data.yaml               # YOLO configuration
├── dataset/                # Prepared dataset
│   ├── train/
│   ├── val/
│   └── test/
└── runs/                   # Training outputs
    └── classify/
        └── skin_diseases/
            ├── weights/
            │   └── best.pt # Trained model
            ├── confusion_matrix.png
            └── classification_report.txt
```

## Model Files

- **Best Model**: `runs/classify/runs/classify/skin_diseases/weights/best.pt` (9.8MB)
- **Confusion Matrix**: `runs/classify/runs/classify/skin_diseases/confusion_matrix.png`
- **Classification Report**: `runs/classify/runs/classify/skin_diseases/classification_report.txt`

## Training Details

- **Epochs**: 50
- **Batch Size**: 16
- **Image Size**: 224x224
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Device**: GPU (Metal Performance Shaders)

## License

Dataset: Apache 2.0 (Kaggle)
