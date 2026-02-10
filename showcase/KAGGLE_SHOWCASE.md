# 🔬 YOLOv8 Skin Diseases Classification - Live Inference Demo

> **Model Performance**: 99.34% Test Accuracy | YOLOv8 Small | 9.8MB

This notebook demonstrates real-time predictions from our trained YOLOv8 model on skin disease images.

## 📊 Demo Statistics

| Metric | Value |
|--------|-------|
| **Total Predictions** | 15 |
| **Correct Predictions** | 15 |
| **Demo Accuracy** | 100.00% |
| **Average Confidence** | 99.11% |

---

## 🎯 Predictions by Disease Type

### 1️⃣ Acne

<table>
<tr>
<td width="33%">

![acne sample 1](images/acne_ac46_JPG.rf.f4062b9b3d00fd4119a5263502cd2bc1.jpg)

**Prediction**: Acne  
**Confidence**: 99.99% ✅

</td>
<td width="33%">

![acne sample 2](images/acne_ac92_JPG.rf.e5cb993c4db20f18552a521b13555524.jpg)

**Prediction**: Acne  
**Confidence**: 99.63% ✅

</td>
<td width="33%">

![acne sample 3](images/acne_n1_JPG.rf.be67c87bc9cb96a18049893c8fe9f7c2.jpg)

**Prediction**: Acne  
**Confidence**: 98.55% ✅

</td>
</tr>
</table>

---

### 2️⃣ Eksim (Eczema)

<table>
<tr>
<td width="33%">

![eksim sample 1](images/eksim_ek23_jpg.rf.4ede4325eea1190e87fb1760ffe165d4.jpg)

**Prediction**: Eksim  
**Confidence**: 99.77% ✅

</td>
<td width="33%">

![eksim sample 2](images/eksim_ek28_JPG.rf.509f6556f37257b12fbfb0c7f53207d1.jpg)

**Prediction**: Eksim  
**Confidence**: 94.83% ✅

</td>
<td width="33%">

![eksim sample 3](images/eksim_ek48_JPG.rf.4130c0a524d50849e0b4c99e7983882a.jpg)

**Prediction**: Eksim  
**Confidence**: 100.00% ✅

</td>
</tr>
</table>

---

### 3️⃣ Herpes

<table>
<tr>
<td width="33%">

![herpes sample 1](images/herpes_he20_jpg.rf.58756866f57add5b1d132a996a1a62f7.jpg)

**Prediction**: Herpes  
**Confidence**: 100.00% ✅

</td>
<td width="33%">

![herpes sample 2](images/herpes_he51_jpg.rf.23e882e9f8b841e96651a2e7bb5c2f91.jpg)

**Prediction**: Herpes  
**Confidence**: 100.00% ✅

</td>
<td width="33%">

![herpes sample 3](images/herpes_he97_jpg.rf.bb445e55b5147a41155e5813e070ba99.jpg)

**Prediction**: Herpes  
**Confidence**: 99.97% ✅

</td>
</tr>
</table>

---

### 4️⃣ Panu (Tinea Versicolor)

<table>
<tr>
<td width="33%">

![panu sample 1](images/panu_pa22_jpg.rf.9deeef41a9f95328d9bca7de9d4ecf49.jpg)

**Prediction**: Panu  
**Confidence**: 96.08% ✅

</td>
<td width="33%">

![panu sample 2](images/panu_pa1_jpeg.rf.43de2cddbae72e1c4b20d9164c0ddfcf.jpg)

**Prediction**: Panu  
**Confidence**: 99.43% ✅

</td>
<td width="33%">

![panu sample 3](images/panu_pa32_jpg.rf.d135c3be0a3939094887de1a97c2ad60.jpg)

**Prediction**: Panu  
**Confidence**: 100.00% ✅

</td>
</tr>
</table>

---

### 5️⃣ Rosacea

<table>
<tr>
<td width="33%">

![rosacea sample 1](images/rosacea_ro77_jpg.rf.8e77cdea1033249fb3dfbb28a25df849.jpg)

**Prediction**: Rosacea  
**Confidence**: 100.00% ✅

</td>
<td width="33%">

![rosacea sample 2](images/rosacea_ro47_jpg.rf.6af3b52ffa6adc3459debeb4cfba6eac.jpg)

**Prediction**: Rosacea  
**Confidence**: 100.00% ✅

</td>
<td width="33%">

![rosacea sample 3](images/rosacea_ro49_jpeg.rf.3a5c1a225f3ae9b2c1e04951d94aa383.jpg)

**Prediction**: Rosacea  
**Confidence**: 99.40% ✅

</td>
</tr>
</table>

---

## 🤖 Model Details

| Specification | Details |
|--------------|---------|
| **Architecture** | YOLOv8 Small Classification |
| **Framework** | Ultralytics YOLOv8 |
| **Training Images** | 1,044 images (5 classes) |
| **Validation Images** | 298 images |
| **Test Images** | 152 images |
| **Test Accuracy** | **99.34%** |
| **Model Size** | 9.8 MB |
| **Input Size** | 224×224 pixels |
| **Training Epochs** | 50 |
| **Optimizer** | AdamW |

## 📈 Per-Class Performance

| Class | Precision | Recall | F1-Score | Test Samples |
|-------|-----------|--------|----------|--------------|
| **Acne** | 100.00% | 100.00% | 100.00% | 30 |
| **Eksim** | 100.00% | 96.77% | 98.36% | 31 |
| **Herpes** | 96.77% | 100.00% | 98.36% | 30 |
| **Panu** | 100.00% | 100.00% | 100.00% | 31 |
| **Rosacea** | 100.00% | 100.00% | 100.00% | 30 |

## 🚀 Quick Start

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Run inference
results = model('path/to/skin_image.jpg')

# Get prediction
pred_class = results[0].probs.top1
confidence = results[0].probs.top1conf.item()

classes = ["acne", "eksim", "herpes", "panu", "rosacea"]
print(f"Prediction: {classes[pred_class]} ({confidence:.2%})")
```

## 📦 Dataset

- **Source**: [Kaggle - Classification of Skin Diseases](https://www.kaggle.com/datasets/sponishflea/classification-of-skin-diseases)
- **License**: Apache 2.0
- **Total Images**: 1,494
- **Classes**: 5 (Acne, Eksim, Herpes, Panu, Rosacea)

## 🎓 Training Configuration

```yaml
Model: yolov8s-cls.pt
Epochs: 50
Batch Size: 16
Image Size: 224x224
Optimizer: AdamW
Learning Rate: 0.001
Device: GPU (Metal Performance Shaders)
Early Stopping: 10 epochs patience
```

## 📝 Notes

- All predictions shown achieved 100% accuracy on these samples
- Model demonstrates excellent generalization across all disease types
- High confidence scores (94.83% - 100.00%) indicate robust feature learning
- Minimal confusion between visually similar conditions

---

**Built with** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | **Dataset** [Kaggle](https://www.kaggle.com/datasets/sponishflea/classification-of-skin-diseases)
