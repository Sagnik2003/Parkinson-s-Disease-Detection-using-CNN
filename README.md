# 🧠 Parkinson's Disease Detection from Hand-Drawn Patterns

This project implements a shallow Convolutional Neural Network (CNN) to classify hand-drawn spirals, circles, and meanders from individuals with Parkinson’s Disease (PD) and healthy controls. To address the limited dataset size, geometric and photometric data augmentations—such as rotation, scaling, and brightness adjustments—were applied. These enhancements helped the CNN effectively identify motor control impairments linked to PD, achieving promising classification results.

---

## 📁 Dataset

- The original and augmented dataset is available on **Kaggle**:  
  🔗 [Link to Kaggle Dataset](#) *(replace with actual URL)*

It contains:
- Original drawings (spirals, circles, meanders)
- Augmented images using geometric and photometric techniques

---

## 🧐 Model Architecture

- A **shallow CNN** architecture with fewer layers to reduce overfitting on small data
- SVG and PNG images of the architecture are included in the repo:
  - `model_architecture.svg`
  - `model_architecture.png`

---

## 🧪 Pretrained Models

Ready-to-use PyTorch `.pth` files are available in the `saved_models/` directory:
- `best_model0.9677.pth` (high validation accuracy)
- Can be loaded and used for inference via the `prediction.py` script

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/parkinsons-detection.git
cd parkinsons-detection
pip install -r requirements.txt
