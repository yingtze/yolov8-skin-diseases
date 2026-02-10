# Kaggle Showcase - Quick Guide

## 🚀 Running on Kaggle/Colab

### Step 1: Download the Dataset

**Before running any scripts**, you must download the dataset.

#### Option A: Automated Download (Recommended)

```python
# Install kagglehub
!pip install kagglehub

# Run the download script
!python3 download_dataset.py
```

The script will automatically:
- Download the dataset
- Set up the `train/` folder with 5 class subdirectories
- Verify the structure

#### Option B: Manual Download

```python
# Install Kaggle API
!pip install kaggle

# Upload your kaggle.json file to the notebook, then:
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download and extract dataset
!kaggle datasets download sponishflea/classification-of-skin-diseases
!unzip -q classification-of-skin-diseases.zip

# Verify the 'train' folder exists
!ls -d train
```

### Step 2: Clone/Upload This Repository

```python
# Option A: Clone from GitHub
!git clone https://github.com/YOUR_USERNAME/yolov8-skin-disease-classifier.git
%cd yolov8-skin-disease-classifier

# Option B: Upload files manually
# Upload all .py files and data.yaml to your notebook
```

### Step 3: Install Dependencies

```python
!pip install ultralytics opencv-python pillow matplotlib scikit-learn seaborn
```

### Step 4: Run the Pipeline

```python
# Prepare dataset
!python3 prepare_dataset.py

# Train model (takes ~15-30 minutes)
!python3 train_yolo.py

# Evaluate model
!python3 test_model.py

# Generate showcase
!python3 create_showcase.py
```

---

## 📁 What's in the Showcase

The `showcase/` directory (168KB total) contains everything you need for a Kaggle demonstration:

### Files
- **KAGGLE_SHOWCASE.md** - Professional showcase with tables and visual layout (258 lines)
- **INFERENCE_SHOWCASE.md** - Detailed results with top-3 predictions (166 lines)
- **README.md** - Directory documentation
- **images/** - 15 sample images (3 per disease class)

### Sample Images Included
- ✅ 3 Acne samples
- ✅ 3 Eksim (Eczema) samples
- ✅ 3 Herpes samples
- ✅ 3 Panu (Tinea Versicolor) samples
- ✅ 3 Rosacea samples

## 🎯 Demo Results
- **100% accuracy** on all 15 showcase samples
- Confidence scores: 94.83% - 100.00%
- All predictions correct with high confidence

## 📤 How to Upload to Kaggle

### Option 1: As a Kaggle Notebook
1. Create a new Kaggle notebook
2. Upload the `showcase/` directory as data
3. Copy content from `KAGGLE_SHOWCASE.md` into markdown cells
4. Images will display automatically

### Option 2: As a Dataset
1. Create a new Kaggle dataset
2. Upload the entire `showcase/` directory
3. Add `KAGGLE_SHOWCASE.md` as the dataset description

### Option 3: In Your Code Notebook
1. Add this to your existing training notebook
2. Upload `showcase/` as supplementary data
3. Reference images in markdown cells

## 🖼️ Image Paths

All images use relative paths:
```markdown
![description](images/acne_sample.jpg)
```

This works automatically when uploaded to Kaggle.

## 🔄 Regenerating the Showcase

To create a new showcase with different samples:

```bash
cd /Users/kharismagp/trainML
python3 create_showcase.py
```

This will:
- Randomly select 3 new images per class
- Run fresh predictions
- Update all markdown files
- Copy new images to showcase/images/

## 📊 What Makes This Showcase Great

1. **Visual Appeal** - Side-by-side image comparisons in tables
2. **Real Predictions** - Actual model outputs, not mock data
3. **High Confidence** - Demonstrates model reliability
4. **Portable** - Self-contained directory (168KB)
5. **Professional** - Clean formatting suitable for Kaggle

## 💡 Tips for Kaggle

- Use `KAGGLE_SHOWCASE.md` for the main presentation
- Use `INFERENCE_SHOWCASE.md` for detailed technical analysis
- Mention the 99.34% test accuracy prominently
- Highlight the 100% demo accuracy
- Include the model size (9.8MB) to show efficiency

## 🎨 Customization

You can edit `create_showcase.py` to:
- Change `SAMPLES_PER_CLASS` (default: 3)
- Modify the markdown formatting
- Add more statistics
- Include different visualizations

---

**Ready to upload!** The showcase directory is completely self-contained and portable.
