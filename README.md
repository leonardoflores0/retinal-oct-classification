# Retinal OCT Image Classification using DenseNet121 and MONAI

A deep learning pipeline for multi-class classification of Optical Coherence Tomography (OCT) retinal images. Built using **PyTorch** and **MONAI** (Medical Open Network for AI), this project implements data preprocessing, deterministic experimental tracking, 2D DenseNet121 transfer learning, and performance evaluation.

---

## Project Overview

Optical Coherence Tomography (OCT) is a non-invasive imaging test used to capture cross-sectional scans of the retina. Automated classification of OCT scans assists ophthalmologists in the early detection and diagnosis of retinal pathologies (such as CNV, DME, and Drusen).

### Key Objectives
* **Domain-Specific Preprocessing**: Utilize MONAI transforms for intensity scaling (`ScaleIntensity`), channel alignment (`EnsureChannelFirst`), and spatial standardization (`Resize`).
* **Deterministic Training**: Guarantee experiment reproducibility across data splits and weight initialization using global seed control (`set_determinism(seed=0)`).
* **Robust Neural Architecture**: Train a 2D DenseNet121 network adapted for 1-channel grayscale medical image inputs.
* **Validation & Metrics**: Track per-epoch validation loss, Accuracy, and multi-class Area Under the ROC Curve (ROC-AUC) with automated best-model checkpointing.

---

## Environment Setup & Dependencies

### Prerequisites
* **Python**: Version 3.8 or higher
* **Compute**: NVIDIA GPU with CUDA acceleration (e.g., Google Colab T4 GPU instance)

### Core Dependencies
```bash
torch >= 2.0.0
torchvision >= 0.15.0
monai >= 1.2.0
numpy >= 1.22.0
matplotlib >= 3.5.0
pillow >= 9.0.0
scikit-learn >= 1.0.0
tensorboard >= 2.10.0
```

---

## Data Pipeline & Augmentation

### 1. Data Ingestion & Extraction
* The dataset is loaded from a compressed archive `ZhangLabData.zip` containing categorized retinal OCT scans.
* It is extracted into local runtime storage `content/data_dir` with directory-derived target class labels `CNV`, `DME`, `DRUSEN`, `NORMAL`.

### 2. Dataset Partitioning
The dataset is partitioned into three mutually exclusive subsets using shuffled indices:
* **Training Set**: 80%
* **Validation Set**: 10%
* **Testing Set**: 10%

### 3. Preprocessing & Data Augmentations
Medical imaging data requires explicit handling to ensure channel consistency and intensity normalization.

* **Training Transformations (`train_transforms`)**:
  - `LoadImage(image_only=True)`: Loads raw image array.
  - `EnsureChannelFirst()`: Standardizes tensor shape to `(1, H, W)` for single-channel grayscale OCT images.
  - `ScaleIntensity()`: Scales raw pixel intensities into a normalized range `[0, 1]`.
  - `Resize((224, 224))`: Standardizes spatial dimensions to 224x224 pixels.
  - `RandRotate(range_x=π/12, prob=0.5)`: Stochastic rotation up to 15 degrees.
  - `RandFlip(spatial_axis=1, prob=0.5)`: Horizontal flipping.
  - `RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5)`: Spatial scaling/zooming.

* **Validation & Test Transformations (`val_transforms`)**:
  - `LoadImage` -> `EnsureChannelFirst` -> `ScaleIntensity` -> `Resize`.

---

## Model Architecture & Training Protocol

* **Backbone**: MONAI 2D DenseNet121 (`spatial_dims=2`, `in_channels=1`,`out_channels=4`).
* **Loss Function**: PyTorch `CrossEntropyLoss()`.
* **Optimizer**: Adam optimizer with a initial learning rate of `1e-5`.
* **Batch Size**: 64 with multi-threaded loading `num_workers=2`.
* **Epoch Count**: `max_epochs = 4` with validation checks `val_interval = 1`
* **Evaluation Frequency**: Validation loss, accuracy, and ROC-AUC are evaluated after every epoch.
* **Model Checkpointing**: Automated saving of model weights `best_metric_model.pth` whenever validation ROC-AUC reaches a new peak score.

---

## Results & Visualizations

### TensorBoard Logging
Step-by-step training loss and validation accuracy metrics are written to the TensorBoard event logs stored in the `runs/` directory.

To inspect training dynamics locally:
```bash
tensorboard --logdir=runs/
```
Then open `http://localhost:6006/` in your browser.
### Performance Evaluation
1. **Training & Validation Trajectory**: Plots generated via Matplotlib comparing epoch average loss against validation ROC-AUC.
2. **Holdout Test Set Classification**: The saved checkpoint `best_metric_model.pth` is reloaded to evaluate unseen test samples, generating a per-class `scikit-learn` classification report evaluating Precision, Recall, F1-Score, and overall Accuracy.

---

## Acknowledgments & Citations

**MONAI** 

Medical Open Network for AI framework.

**Zhang Lab Dataset** 

Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images", Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3