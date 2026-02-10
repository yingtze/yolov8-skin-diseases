# GitHub Upload Guide

## What's Included in Git

The following files will be committed to GitHub:
- ✅ `README.md` - Project documentation
- ✅ `prepare_dataset.py` - Dataset preparation script
- ✅ `train_yolo.py` - Training script
- ✅ `test_model.py` - Evaluation script
- ✅ `data.yaml` - YOLO configuration
- ✅ `.gitignore` - Git ignore rules
- ✅ `.gitattributes` - Git attributes for line endings

## What's Excluded (via .gitignore)

The following large/sensitive files are excluded:
- ❌ `train/` - Original dataset (12MB zip + extracted files)
- ❌ `dataset/` - Prepared dataset (~1,500 images)
- ❌ `runs/` - Training outputs and model weights
- ❌ `*.pt` - Model checkpoint files
- ❌ `*.log`, `*.txt` - Log files
- ❌ `classification-of-skin-diseases.zip` - Downloaded dataset

## Steps to Upload to GitHub

### 1. Create a new repository on GitHub
Go to https://github.com/new and create a new repository (e.g., `yolo-skin-diseases`)

### 2. Add and commit files
```bash
cd /Users/kharismagp/trainML
git add .
git commit -m "Initial commit: YOLOv8 skin diseases classification"
```

### 3. Connect to GitHub and push
```bash
# Replace YOUR_USERNAME and YOUR_REPO with your actual GitHub username and repo name
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Optional: Upload Model Weights Separately

Since model weights are large (9.8MB), you have options:

### Option 1: Git LFS (Large File Storage)
```bash
# Install Git LFS
brew install git-lfs  # macOS
git lfs install

# Track model files
git lfs track "*.pt"
git add .gitattributes
git add runs/classify/runs/classify/skin_diseases/weights/best.pt
git commit -m "Add trained model weights"
git push
```

### Option 2: GitHub Releases
1. Create a release on GitHub
2. Upload `best.pt` as a release asset
3. Update README with download link

### Option 3: External Hosting
- Upload to Google Drive, Dropbox, or Hugging Face
- Add download link to README

## Repository Size

**Without model weights**: ~50KB (just code and config)
**With model weights**: ~10MB (if using Git LFS)

## Recommended: Add a License

Create a `LICENSE` file. Popular choices:
- MIT License (permissive)
- Apache 2.0 (permissive with patent grant)
- GPL-3.0 (copyleft)

## Next Steps After Upload

1. Add repository topics on GitHub: `yolo`, `computer-vision`, `skin-diseases`, `classification`, `pytorch`
2. Add a repository description
3. Enable GitHub Actions for CI/CD (optional)
4. Add badges to README (optional)
