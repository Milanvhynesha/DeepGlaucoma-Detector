# 👁️ DeepGlaucoma-Detector: Automated Glaucoma Detection Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![MONAI](https://img.shields.io/badge/MONAI-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Overview

DeepGlaucoma-Detector is a deep learning-based system that automatically detects glaucoma from eye fundus images. Built with PyTorch and MONAI (Medical Open Network for AI), it uses a DenseNet121 architecture trained on clinical data to distinguish between glaucomatous and healthy eyes with high accuracy.

**Why Glaucoma Detection Matters:** Glaucoma is the second leading cause of blindness worldwide. Early detection through automated screening can help prevent vision loss and improve patient outcomes.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Accurate Detection** | DenseNet121 model achieves high accuracy in glaucoma detection |
| 📊 **Complete Pipeline** | End-to-end workflow from data loading to model deployment |
| 🏥 **Medical Imaging Focus** | Built with MONAI, specifically designed for healthcare AI |
| 🔍 **Model Interpretability** | Grad-CAM visualization shows which image regions influence decisions |
| 📈 **Batch Processing** | Analyze hundreds of images simultaneously |
| 🚀 **Deployment Ready** | Flask API, Docker containerization options |

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~90% |
| AUC-ROC | ~0.95 |
| Sensitivity | ~88% |
| Specificity | ~92% |

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch, MONAI, DenseNet121
- **Image Processing**: OpenCV, Pillow
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Flask, Docker

## 🚀 Quick Start

```python
from src.predict import GlaucomaPredictor

predictor = GlaucomaPredictor('models/best_model.pth')
result = predictor.predict('eye_image.jpg')
print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 📦 Installation
```
git clone https://github.com/Milanvhynesha/DeepGlaucoma-Detector.git
cd DeepGlaucoma-Detector
pip install -r requirements.txt
```

## 📄 License
MIT License

## 📧 Contact
Milanvhynesha - GitHub Profile

