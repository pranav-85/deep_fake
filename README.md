# Deepfake Detection Using Optical Flow and CNN-Based Architectures

## Introduction

Deepfakes leverage AI to manipulate videos by altering faces, often convincingly. This project aims to detect deepfakes using both frame-level and motion-based analysis. We evaluate a 3D CNN with optical flow, and compare it against state-of-the-art image classifiers like ResNet152, Vision Transformer (ViT), and XceptionNet.

---

## 📂 Dataset

- **Source**: Celeb-DF dataset
- **Training Samples**: 5168
  - Real: 742
  - Fake: 4426
- **Preprocessing**:
  - Videos are split into frames.
  - Frame folders are labeled in the format `id0_id2_0003`.
  - Class balance is ensured by uniform sampling.

---

## 🧠 Methodology

### 1. Frame-Based Classification

Implemented using PyTorch and [timm](https://github.com/huggingface/pytorch-image-models):

- **ResNet152**: Deep CNN with skip connections.
- **ViT (vit_b_16)**: Transformer treating images as patch sequences.
- **XceptionNet**: CNN with depthwise separable convolutions.

**Training Setup**:
- Optimizer: Adam
- Loss: Cross-Entropy
- Epochs: 10
- Libraries: tqdm, scikit-learn

### 2. Optical Flow + 3D CNN

Designed but **not trained due to resource constraints**.

- Motion estimation via Farneback's optical flow.
- Input: 9 motion maps (u and v vectors from 10 consecutive frames).
- Model: 4 Conv3D layers + ReLU + MaxPool3D + fully connected layers.

---

## 📈 Metrics & Evaluation

- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, EER
- **Visualization**: Confusion matrix, ROC curve, Precision-Recall curve

---

## 🧪 Results

| Model        | Accuracy | Precision | AUC  | EER  |
|--------------|----------|-----------|------|------|
| ResNet152    | 94.1%    | 0.91      | 0.98 | 0.05 |
| ViT (vit_b_16) | 91.3%  | 0.89      | 0.96 | 0.09 |
| XceptionNet  | 93.9%    | 0.91      | 0.97 | 0.07 |

---

## ✅ Conclusion

- **ResNet152**: Best performance overall.
- **XceptionNet**: Competitive results, low error.
- **ViT**: Slightly lower performance.
- **Optical Flow + 3D CNN**: Promising direction, needs further computation and training.

---

## 🚧 Future Work

- Train and evaluate the 3D CNN + Optical Flow pipeline.
- Experiment with temporal transformers or hybrid models.
- Evaluate performance on unseen datasets for generalization.

---

## 🛠️ Technologies Used

- Python, PyTorch, timm
- Scikit-learn, OpenCV (for optical flow)
- Celeb-DF dataset

---