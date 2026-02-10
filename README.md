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

### Prerequisites

Install required packages:
```bash
pip install ultralytics kaggle opencv-python pillow matplotlib scikit-learn seaborn
```

### Step 1: Download Dataset

**Important:** You must download the dataset before running any scripts.

#### Option A: Using Kaggle API (Recommended)

1. Get your Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token" to download `kaggle.json`

2. Set up Kaggle API:
   ```bash
   # Create kaggle directory
   mkdir -p ~/.kaggle
   
   # Copy your kaggle.json to the directory
   cp /path/to/kaggle.json ~/.kaggle/
   
   # Set permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Download and extract the dataset:
   ```bash
   # Download dataset
   kaggle datasets download sponishflea/classification-of-skin-diseases
   
   # Extract
   unzip classification-of-skin-diseases.zip
   
   # Verify the 'train' folder exists
   ls -d train
   ```

#### Option B: Manual Download

1. Visit [Kaggle Dataset Page](https://www.kaggle.com/datasets/sponishflea/classification-of-skin-diseases)
2. Click "Download" button
3. Extract the zip file in the project directory
4. Ensure you have a `train/` folder with 5 subdirectories (acne, eksim, herpes, panu, rosacea)

### Step 2: Prepare Dataset

After downloading, prepare the dataset for YOLO training:
```bash
python3 prepare_dataset.py
```

This will create a `dataset/` directory with train/val/test splits.

### Step 3: Train Model

```bash
python3 train_yolo.py
```

Training will take approximately 15-30 minutes depending on your hardware.

### Step 4: Evaluate Model

```bash
python3 test_model.py
```

This generates confusion matrix and classification report.

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
