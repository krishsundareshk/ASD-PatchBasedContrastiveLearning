# ASTRA: Attention-based Self-supervised Transformative Representation for Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A patch-based contrastive learning framework for anomaly detection in industrial audio, leveraging RGB spectrogram representations and attribute-conditioned attention pooling.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [1. Audio to RGB Conversion](#1-audio-to-rgb-conversion)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

ASTRA implements a novel approach to industrial anomaly detection by:
- Converting audio signals to RGB spectrograms using mel-scale transformations
- Employing patch-based contrastive learning (SimCLR-style) with ResNet34 backbone
- Utilizing attribute-conditioned attention pooling for robust feature aggregation
- Supporting multi-domain evaluation with separate source/target domain modeling
- Implementing multi-centroid Mahalanobis distance with PCA dimensionality reduction

**Supported Machine Types:** `ToyCar`, `ToyTrain`, `bearing`, `valve`, `fan`, `gearbox`, `slider`

---

## ✨ Features

- **RGB Spectrogram Generation**: Converts WAV files to 224×224 RGB spectrograms using customizable colormaps
- **Patch-Based Learning**: Extracts and processes 32×32 patches with configurable stride
- **Attribute Conditioning**: Incorporates machine attributes (e.g., model ID, operational parameters) into attention mechanism
- **Multi-Centroid Modeling**: Uses K-means clustering with per-cluster Mahalanobis distances for robust anomaly scoring
- **Ensemble Scoring**: Combines Mahalanobis and cosine distance metrics with z-score normalization
- **Domain Adaptation**: Separate models for source and target domains
- **Automatic Checkpointing**: Saves training progress with configurable retention policy

---

## 📦 Requirements

### Core Dependencies

```
Python >= 3.8
torch >= 2.0.0
torchvision >= 0.15.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
librosa >= 0.10.0
matplotlib >= 3.5.0
Pillow >= 9.0.0
tqdm >= 4.62.0
```

### Hardware Requirements

- **GPU**: CUDA-capable GPU with 8GB+ VRAM recommended (RTX 3070 or better)
- **RAM**: 16GB+ system memory
- **Storage**: ~50GB for dataset and checkpoints

---

## 🚀 Installation

### Option 1: pip install

```bash
# Clone the repository
git clone https://github.com/yourusername/astra-anomaly-detection.git
cd astra-anomaly-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda environment

```bash
# Create conda environment
conda create -n astra python=3.8
conda activate astra

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install librosa matplotlib pandas scikit-learn tqdm Pillow
```

---

## 📂 Dataset Preparation

### Expected Directory Structure

```
Audio/
├── ToyCar/
│   ├── train/
│   │   ├── normal_id_01_00000000.wav
│   │   ├── normal_id_01_00000001.wav
│   │   └── ...
│   ├── supplemental/
│   │   └── ...
│   ├── test/
│   │   ├── normal_id_01_00000000.wav
│   │   ├── anomaly_id_01_00000000.wav
│   │   └── ...
│   └── attributes_00.csv
├── fan/
│   └── ...
└── ...
```

### Attributes CSV Format

The `attributes_00.csv` should contain machine-specific metadata:

```csv
filename,model_id,speed,load
normal_id_01_00000000.wav,1,1000,50
normal_id_01_00000001.wav,1,1200,60
...
```

**Note**: Categorical attributes will be automatically encoded using ordinal encoding.

---

## 💻 Usage

### 1. Audio to RGB Conversion

Convert all WAV files to RGB spectrograms:

```bash
python convert_rgb.py
```

**Configuration Options** (edit in `convert_rgb.py`):

```python
# Spectrogram parameters
sr=16000          # Sampling rate
n_fft=1024        # FFT window size
hop_length=512    # Hop length for STFT
n_mels=128        # Number of mel bands
fmin=20           # Minimum frequency
fmax=8000         # Maximum frequency
cmap_name="plasma"  # Colormap: plasma, viridis, magma, inferno
```

**Output**: Creates `trainRGB/`, `supplementalRGB/`, `testRGB/` folders with 224×224 PNG files.

---

### 2. Training

Train the joint model across all machine types:

```bash
python train_joint.py
```

**Key Parameters** (in `train_joint.py`):

```python
ROOT_DIR = "training_data"          # Path to dataset root
CHECKPOINT_DIR = "checkpoints_dino"  # Checkpoint save directory
BATCH_SIZE = 32                     # Batch size
EPOCHS = 100                        # Number of epochs
LEARNING_RATE = 2e-4                # Initial learning rate
TEMPERATURE = 0.1                   # NT-Xent temperature
MAX_PATCHES = 64                    # Max patches per image
STRIDE = 16                         # Patch extraction stride
```

**Features**:
- ✅ Automatic checkpoint resumption
- ✅ Retains only last 5 checkpoints to save disk space
- ✅ ReduceLROnPlateau scheduler with patience
- ✅ Multi-GPU support (DataParallel compatible)

**Training Output**:

```
🚀 Starting training from scratch
[Epoch 1/100]: 100%|████████| 1234/1234 [05:23<00:00, loss=2.3456]
Epoch 1 avg loss: 2.3456
...
✅ Joint training complete. Checkpoints in: checkpoints_dino
```

---

### 3. Evaluation

Evaluate trained model on test set:

```bash
python eval_joint.py
```

**Configuration** (in `eval_joint.py`):

```python
CHECKPOINT_DIR = "checkpoints_dino"  # Path to checkpoints
EVAL_EPOCH = 100                     # Epoch to evaluate
BATCH_SIZE = 256                     # Evaluation batch size

# Multi-centroid modeling
CLUSTER_K = 5                        # Number of clusters
COV_TYPE = "lw"                      # Covariance estimator: lw, oas, empirical, diag
USE_PCA = True                       # Apply PCA before Mahalanobis
PCA_VARIANCE = 0.98                  # Retained variance ratio

# Score ensemble
USE_COSINE = True                    # Enable cosine distance
W_MAHA = 0.7                         # Mahalanobis weight
W_COS = 0.3                          # Cosine weight

# Thresholding
USE_TARGET_FPR = True                # Use FPR-based threshold
TARGET_FPR = 0.05                    # Target false positive rate
```

**Evaluation Metrics**:
- AUC (Area Under ROC Curve)
- pAUC (Partial AUC with max_fpr=0.1)
- Accuracy, F1-score, Precision, Recall (at optimal threshold)

**Output Example**:

```
=== ToyCar ===
  src: K=5, cov=lw, PCA=True(0.98), thr=2.134
  tgt: K=5, cov=lw, PCA=True(0.98), thr=2.087
  source AUC=0.9234  pAUC(≤0.1)=0.8567  |  Op: Acc=0.9123 F1=0.8901 P=0.8734 R=0.9089
  target AUC=0.9456  pAUC(≤0.1)=0.8823  |  Op: Acc=0.9334 F1=0.9112 P=0.9001 R=0.9234
...
✅ Done.
```

---

## 🏗️ Model Architecture

### Overall Pipeline

```
Audio → RGB Spectrogram → Patch Extraction → ResNet34 Encoder → 
Projection Head → Attention Pooling → Attribute Fusion → 
Contrastive Loss (Training) / Anomaly Score (Inference)
```

### Components

#### 1. **ResNet34 Encoder**
- Pretrained on ImageNet
- Outputs 512-dimensional features per patch
- Classification head removed

#### 2. **Projection Head**
- 512 → 512 (ReLU) → 128
- Projects backbone features to embedding space

#### 3. **Attention Pooling**
- Attribute-conditioned attention mechanism
- Learnable attention weights per patch
- Output: single pooled embedding per image

#### 4. **Attribute Fusion**
- Encodes machine attributes via MLP: `attr_dim → 32 → 128`
- Early fusion: concatenates with pooled embedding
- Final dimension: `256` (128 pooled + 128 attributes)

#### 5. **Loss Function**
- NT-Xent (Normalized Temperature-scaled Cross Entropy)
- Contrastive learning between augmented views
- Temperature: 0.1

---

## ⚙️ Configuration

### Dataset Configuration

Edit `astra_attn_patch_dataset.py`:

```python
ALL_MACHINE_TYPES = ["ToyCar", "ToyTrain", "bearing", "valve", "fan", "gearbox", "slider"]
USE_ATTR = {"fan", "ToyCar", "valve", "gearbox"}  # Machines with attributes
```

### Augmentation Pipeline

Training augmentations (in `ASTRA_AttnPatchRGBDataset`):

```python
transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Hyperparameter Tuning

Recommended ranges for tuning:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `BATCH_SIZE` | 32 | 16-128 | Larger = more stable gradients |
| `LEARNING_RATE` | 2e-4 | 1e-5 to 1e-3 | Lower for fine-tuning |
| `TEMPERATURE` | 0.1 | 0.05-0.5 | Lower = harder negatives |
| `MAX_PATCHES` | 64 | 32-128 | Memory vs. detail trade-off |
| `CLUSTER_K` | 5 | 3-10 | Number of normal clusters |
| `W_MAHA` / `W_COS` | 0.7 / 0.3 | — | Score ensemble weights |

---

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `BATCH_SIZE` in `train_joint.py`
- Decrease `MAX_PATCHES`
- Use gradient accumulation

**2. Slow Training**
- Increase `num_workers` in DataLoader (default: 4)
- Enable mixed precision training (add AMP)
- Use smaller image size (224 → 128)

**3. Poor Convergence**
- Increase `EPOCHS` (try 200+)
- Adjust `LEARNING_RATE` (try 1e-4 or 5e-4)
- Verify data augmentation isn't too aggressive

**4. Low Test AUC**
- Increase `CLUSTER_K` for more fine-grained modeling
- Try different `COV_TYPE` (lw, oas, empirical)
- Adjust `W_MAHA` / `W_COS` weights
- Enable/disable PCA dimensionality reduction

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{astra2024,
  title={ASTRA: Attention-based Self-supervised Transformative Representation for Anomaly Detection},
  author={Your Name},
  booktitle={Proceedings of...},
  year={2024}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).

---

## 🙏 Acknowledgments

- ResNet34 pretrained weights from [torchvision](https://pytorch.org/vision/stable/models.html)
- SimCLR framework inspiration
- Industrial audio datasets from [DCASE Challenge](http://dcase.community/)

---

**Last Updated**: December 2024
