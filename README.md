# 🔬 Eye Disease Classification using CNN (PyTorch)

A Convolutional Neural Network built from scratch to classify retinal fundus images into 4 eye disease categories — developed as part of the **Computer Vision & Pattern Recognition** coursework at **American International University-Bangladesh (AIUB)**.

---

## 📌 Problem Statement

Automated classification of retinal fundus images to assist in early detection of eye diseases, reducing the dependency on manual diagnosis by ophthalmologists.

---

## 🗂️ Dataset

| Detail | Info |
|--------|------|
| **Name** | AMDNet23 |
| **Source** | [Mendeley Data](https://data.mendeley.com/) |
| **Total Images** | 2,000 fundus images |
| **Classes** | 4 |

### Classes
| Label | Disease |
|-------|---------|
| 0 | Normal |
| 1 | Diabetic Retinopathy |
| 2 | Cataract |
| 3 | Age-related Macular Degeneration (AMD) |

---

## 🏗️ Model Architecture

```
Input (3 × H × W)
    │
    ├── Conv2d → BatchNorm → ReLU → MaxPool
    ├── Conv2d → BatchNorm → ReLU → MaxPool
    ├── Conv2d → BatchNorm → ReLU → MaxPool
    │
    ├── Flatten
    │
    ├── Linear (FC1) → ReLU → Dropout(0.3)
    └── Linear (FC2) → Softmax (4 classes)
```

| Component | Detail |
|-----------|--------|
| Convolutional Layers | 3 |
| Pooling | MaxPooling |
| Fully Connected Layers | 2 |
| Dropout Rate | 0.3 |
| Total Parameters | **2,121,380** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 40 |
| Loss Function | CrossEntropyLoss |

---

## 📊 Results

### Round 1 — No Regularization

| Metric | Value |
|--------|-------|
| Training Accuracy | ~100% |
| Validation Accuracy | ~83% |
| Train-Val Gap | **~17%** ❌ Overfitting |

> The model memorized training images instead of generalizing — a textbook overfitting case.

---

### Round 2 — With Augmentation + Dropout

| Metric | Value |
|--------|-------|
| Training Accuracy | ~87.6% |
| Validation Accuracy | **82.25%** |
| Train-Val Gap | **~5.4%** ✅ Controlled |

> Regularization kept validation performance stable while significantly closing the generalization gap.

---

## 🛠️ Regularization Techniques Used

- ✅ **Random Horizontal Flip** — data augmentation
- ✅ **Random Rotation** — data augmentation
- ✅ **Dropout (p=0.3)** — after the first fully connected layer
- ✅ **Batch Normalization** — after each convolutional layer

---

## 🔍 Visualizations

- **Convolutional Filter Visualization** — to inspect learned low-level features
- **Feature Map Visualization** — to understand what spatial regions the network activates on for each class

---

## 💡 Key Takeaway

> In deep learning, **a small train-validation gap matters more than chasing high training accuracy alone.**  
> A model with 82% validation accuracy and a 5% gap is far more reliable than one with 100% training accuracy but 17% overfitting.

---

## 📁 Repository Structure

```
eye-disease-cnn/
│
├── data/                  # Dataset (not included — see Mendeley link)
├── notebooks/
│   ├── round1_no_regularization.ipynb
│   └── round2_with_regularization.ipynb
├── models/
│   └── cnn_model.py       # Model architecture
├── utils/
│   ├── dataset.py         # Data loading & transforms
│   └── visualize.py       # Filter & feature map plots
├── train.py               # Training script
├── evaluate.py            # Evaluation script
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

### Train the Model

```bash
python train.py --epochs 40 --lr 0.001 --dropout 0.3
```

### Evaluate

```bash
python evaluate.py --model_path checkpoints/best_model.pth
```

---

## 📚 Course

**Computer Vision & Pattern Recognition**  
American International University-Bangladesh (AIUB)

---

## 👤 Author

**Md. Sayed Mamun**  
BSc in Computer Science & Engineering, AIUB  
[LinkedIn](https://linkedin.com/in/md-sayed-mamun-725976276) · [GitHub](https://lnkd.in/duYzuHfU)

---

## 📄 License

This project is for academic purposes. Dataset credits go to the original authors of **AMDNet23** on Mendeley Data.
