# Showcase Directory

This directory contains inference demonstrations for the YOLOv8 skin diseases classification model.

## Files

- **KAGGLE_SHOWCASE.md** - Polished showcase for Kaggle with professional formatting
- **INFERENCE_SHOWCASE.md** - Detailed inference results with top-3 predictions
- **images/** - Sample prediction images (15 images, 3 per class)

## Usage for Kaggle

1. Upload the entire `showcase/` directory to your Kaggle notebook
2. Reference the markdown file in your notebook
3. Images will be displayed inline

## Generating New Showcases

Run the showcase generator from the project root:

```bash
python3 create_showcase.py
```

This will:
- Select 3 random images per class from the test set
- Run inference with the trained model
- Copy images to `showcase/images/`
- Generate both markdown files

## Statistics

- **Total Samples**: 15 (3 per class)
- **Demo Accuracy**: 100%
- **Image Size**: ~7-10KB per image
- **Total Directory Size**: ~120KB
