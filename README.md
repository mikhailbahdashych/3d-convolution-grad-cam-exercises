# Exercise Recognition with 3D Convolutional Neural Networks and GradCAM

A comprehensive implementation of video-based exercise recognition using 3D Convolutional Neural Networks with GradCAM visualization for model interpretability.

---

## Table of Contents

1. [Overview](#overview)
2. [Technical Architecture](#technical-architecture)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Quick Start Guide](#quick-start-guide)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Configuration](#configuration)
9. [Implementation Details](#implementation-details)
10. [Expected Results](#expected-results)

---

## Overview

This project implements a complete pipeline for recognizing exercises from video data. The system uses a 3D Convolutional Neural Network (C3D architecture) to learn spatio-temporal features from video clips and provides interpretability through GradCAM heatmap visualizations.

### Key Features

- 3D CNN (C3D) architecture for video classification
- Focal Loss for handling severe class imbalance
- GradCAM visualization for model interpretability
- Support for multiple devices: CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU
- Comprehensive evaluation metrics and confusion matrix generation
- Modular and extensible codebase

### Use Case

The system is designed for automatic exercise recognition from video footage, capable of classifying 17 different exercise types from short video clips.

---

## Technical Architecture

### Model: C3D (3D Convolutional Network)

**Architecture Overview:**
- Input: RGB video clip of shape (3, 16, 112, 112)
  - 3 color channels
  - 16 temporal frames
  - 112x112 spatial resolution
- 5 convolutional blocks with 3x3x3 kernels
- Batch normalization and ReLU activation
- 3D max pooling for temporal and spatial downsampling
- Global average pooling
- Fully connected classifier
- Output: 17-class predictions

**Model Statistics:**
- Total parameters: 27,797,137 (approximately 27.8 million)
- Trainable parameters: 27,797,137

**Network Details:**
```
Block 1: Conv3D(3->64)   + BatchNorm + ReLU + MaxPool(1,2,2)
Block 2: Conv3D(64->128) + BatchNorm + ReLU + MaxPool(2,2,2)
Block 3: Conv3D(128->256)x2 + BatchNorm + ReLU + MaxPool(2,2,2)
Block 4: Conv3D(256->512)x2 + BatchNorm + ReLU + MaxPool(2,2,2)
Block 5: Conv3D(512->512)x2 + BatchNorm + ReLU + MaxPool(2,2,2)
Global Average Pooling -> Dropout(0.5) -> Linear(512->256) -> ReLU -> Dropout(0.3) -> Linear(256->17)
```

---

## Dataset

### Data Statistics

- Total subjects: 60
- Train/test split: 30 subjects each
- Exercise classes: 17 (labeled as classes 1-16)
- Video format: MP4, 30 FPS
- Original resolution: approximately 400x550 pixels
- Total training clips: 28,413 (with temporal sliding window)
- Total test clips: approximately 23,000

### Data Structure

```
dataset/
├── dataset/
│   ├── anon/              # Anonymized video files (.mp4)
│   ├── mask/              # Segmentation masks (.png)
│   └── skeleton/          # YOLO pose detection outputs
│       ├── yolo/          # Pose videos
│       └── yolo_pose_csv/ # Pose keypoints in CSV format
├── label/                 # Frame-level labels (.csv)
└── split.csv             # Train/test split specification
```

### Label Format

Labels are stored as CSV files with format:
```
frame_number, column1, column2
```
- column2 contains the exercise class: -1 for background, 0-16 for exercises
- One label file per subject

### Class Distribution

The dataset exhibits severe class imbalance:
- Class 1: approximately 64,000 training frames (dominant class)
- Classes 2-16: approximately 9,000-11,000 frames each

---

## Installation

### Prerequisites

- Python 3.12 or higher
- Virtual environment tool (venv)
- uv package manager

### Setup Instructions

1. Clone the repository and navigate to the project directory

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

3. Install dependencies using uv:
```bash
uv add torch torchvision
uv add opencv-python pandas scikit-learn
uv add matplotlib seaborn tensorboard
uv add tqdm pyyaml grad-cam
```

Note: The `decord` package is not available for macOS ARM architecture and is optional. OpenCV is used for video loading instead.

---

## Quick Start Guide

This section provides the recommended sequence for implementing and running the complete pipeline.

### Step 1: Verify Installation

Test that all components are properly installed:

```bash
python scripts/quick_test.py
```

This will verify:
- Dataset loading functionality
- Model initialization
- Forward and backward passes
- Device selection (CPU/MPS/CUDA)

### Step 2: Analyze the Dataset

Explore the dataset statistics and distribution:

```bash
python scripts/data_analysis.py
```

This generates:
- Train/test split statistics
- Class distribution analysis
- Video property statistics
- Visualization plots saved to `outputs/analysis/`

### Step 3: Run Initial Training Test

Perform a quick training test (2 epochs) to verify the pipeline:

```bash
python scripts/train.py --epochs 2 --batch-size 4
```

Expected duration: 5-10 minutes on MPS/GPU

This validates:
- Data loading pipeline
- Model training loop
- Checkpoint saving
- Metrics tracking

### Step 4: Full Training

Launch full training run:

```bash
# Default configuration (100 epochs)
python scripts/train.py

# Custom configuration
python scripts/train.py --epochs 50 --batch-size 8

# Specify device
python scripts/train.py --epochs 50 --device cuda
python scripts/train.py --epochs 50 --device mps
```

Expected duration: 2-4 hours for 50 epochs on MPS/GPU

Training artifacts saved to:
- Checkpoints: `outputs/checkpoints/`
- Logs: `outputs/logs/`
- TensorBoard logs: `outputs/logs/`

### Step 5: Monitor Training Progress

In a separate terminal, launch TensorBoard:

```bash
tensorboard --logdir outputs/logs
```

Access the dashboard at: http://localhost:6006

Metrics tracked:
- Training and validation loss
- Training and validation accuracy
- Per-class accuracy
- Learning rate schedule

### Step 5b: Plot Training Metrics

The training process automatically saves metrics to CSV files after each epoch in `outputs/logs/metrics/`. Generate publication-ready plots from these metrics:

```bash
# Generate all plots with default settings
python scripts/plot_training.py

# Specify custom directories
python scripts/plot_training.py --metrics-dir outputs/logs/metrics --save-dir outputs/plots

# Plot only top 5 classes by accuracy
python scripts/plot_training.py --top-classes 5
```

Generated plots:
- `loss_curves.png` - Training and validation loss over time
- `accuracy_curves.png` - Training and validation accuracy over time
- `learning_rate.png` - Learning rate schedule
- `combined_metrics.png` - All metrics in a single figure including generalization gap
- `per_class_accuracy.png` - Per-class accuracy evolution (all classes)
- `per_class_accuracy_top10.png` - Top 10 performing classes
- `per_class_heatmap.png` - Heatmap showing per-class accuracy over epochs
- `final_class_comparison.png` - Bar chart of final accuracy per class

CSV files saved to `outputs/logs/metrics/`:
- `training_metrics.csv` - Epoch-level loss, accuracy, and learning rate
- `per_class_accuracy.csv` - Per-class accuracy for every epoch

### Step 6: Evaluate Model

After training completes, evaluate on test set:

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model_acc.pth
```

This generates:
- Overall accuracy metrics
- Per-class precision, recall, F1-score
- Confusion matrix (counts and normalized)
- Predictions CSV
- Error analysis

Results saved to: `outputs/results/`

### Step 7: Generate GradCAM Visualizations

Visualize model attention with GradCAM:

```bash
# Visualize 20 random test samples
python scripts/visualize_gradcam.py --checkpoint outputs/checkpoints/best_model_acc.pth --num-samples 20

# Visualize only misclassified samples
python scripts/visualize_gradcam.py --checkpoint outputs/checkpoints/best_model_acc.pth --num-samples 20 --misclassified-only

# Custom visualization settings
python scripts/visualize_gradcam.py \
    --checkpoint outputs/checkpoints/best_model_acc.pth \
    --num-samples 50 \
    --layer block5 \
    --alpha 0.6 \
    --fps 10
```

Outputs for each sample:
- original.mp4: Original video clip
- heatmap.mp4: GradCAM heatmap
- overlay.mp4: Heatmap overlayed on video
- side_by_side.mp4: Original and overlay side-by-side
- metadata.txt: Sample information

Visualizations saved to: `outputs/visualizations/`

---

## Usage

### Training Options

```bash
python scripts/train.py [OPTIONS]

Options:
  --config PATH         Path to configuration YAML file
  --seed INT           Random seed for reproducibility (default: 42)
  --device DEVICE      Device to use: cuda/mps/cpu (auto-detected if not specified)
  --epochs INT         Number of training epochs (default: 100)
  --batch-size INT     Batch size for training (default: 8)
```

### Evaluation Options

```bash
python scripts/evaluate.py [OPTIONS]

Required:
  --checkpoint PATH    Path to model checkpoint (.pth file)

Optional:
  --config PATH        Path to configuration file
  --split SPLIT        Dataset split to evaluate: train/test (default: test)
  --batch-size INT     Batch size for evaluation (default: 16)
  --device DEVICE      Device to use: cuda/mps/cpu
  --save-dir PATH      Directory to save results (default: outputs/results)
```

### GradCAM Visualization Options

```bash
python scripts/visualize_gradcam.py [OPTIONS]

Required:
  --checkpoint PATH    Path to model checkpoint (.pth file)

Optional:
  --config PATH           Path to configuration file
  --split SPLIT           Dataset split: train/test (default: test)
  --num-samples INT       Number of samples to visualize (default: 20)
  --device DEVICE         Device to use: cuda/mps/cpu
  --save-dir PATH         Directory to save visualizations (default: outputs/visualizations)
  --layer LAYER           Target layer for GradCAM: block3/block4/block5 (default: block5)
  --alpha FLOAT           Overlay transparency: 0.0-1.0 (default: 0.5)
  --fps INT               Output video frame rate (default: 10)
  --misclassified-only    Only visualize misclassified samples
```

---

## Project Structure

```
PROJECT/
├── src/                              # Source code
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py                 # Configuration management
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset for video clips
│   │   ├── transforms.py             # Video augmentation transforms
│   │   └── utils.py                  # Video loading utilities
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn3d.py                  # C3D architecture implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop and logic
│   │   └── losses.py                 # Focal Loss implementation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── evaluator.py              # Evaluation pipeline
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── gradcam.py                # GradCAM wrapper for 3D CNN
│   └── utils/
│       ├── __init__.py
│       ├── device.py                 # Device selection (CPU/MPS/CUDA)
│       ├── logging.py                # Logging utilities
│       └── checkpointing.py          # Model save/load utilities
├── scripts/                          # Executable scripts
│   ├── train.py                      # Main training script
│   ├── evaluate.py                   # Model evaluation script
│   ├── visualize_gradcam.py          # GradCAM visualization script
│   ├── data_analysis.py              # Dataset analysis script
│   ├── test_dataset.py               # Dataset loading test
│   ├── test_training_init.py         # Training initialization test
│   └── quick_test.py                 # Quick system test
├── outputs/                          # Generated outputs
│   ├── checkpoints/                  # Model checkpoints (.pth files)
│   ├── logs/                         # Training logs and TensorBoard
│   ├── results/                      # Evaluation results
│   ├── visualizations/               # GradCAM visualizations
│   └── analysis/                     # Dataset analysis plots
├── dataset/                          # Dataset directory (not in repo)
├── .venv/                            # Virtual environment
├── .gitignore                        # Git ignore file
├── pyproject.toml                    # Project dependencies
└── README.md                         # This file
```

---

## Configuration

### Default Configuration

The default configuration is defined in `src/config/config.py`. Key parameters include:

**Data Processing:**
- Clip length: 16 frames
- Temporal stride: 8 frames (for sliding window during training)
- Spatial size: 112x112 pixels
- FPS: 30 (original video framerate)

**Model:**
- Architecture: C3D
- Number of classes: 17
- Dropout: 0.5
- Pretrained weights: False

**Training:**
- Batch size: 8
- Number of epochs: 100
- Number of workers: 4
- Pin memory: True

**Optimizer:**
- Type: AdamW
- Learning rate: 0.0001
- Weight decay: 0.00001

**Learning Rate Scheduler:**
- Type: ReduceLROnPlateau
- Mode: min (reduce on validation loss)
- Factor: 0.5
- Patience: 5 epochs
- Minimum LR: 0.0000001

**Loss Function:**
- Type: Focal Loss
- Gamma: 2.0
- Class weights: Computed automatically from training data

**Data Augmentation (Training Only):**
- Horizontal flip: 50% probability
- Rotation: +/- 10 degrees
- Color jitter: brightness, contrast, saturation, hue
- Normalization: ImageNet mean and std

**Early Stopping:**
- Patience: 10 epochs
- Minimum delta: 0.001

**Checkpointing:**
- Save frequency: Every 10 epochs
- Keep best 3 checkpoints
- Save best model by validation loss
- Save best model by validation accuracy

**Device:**
- Preference order: CUDA > MPS > CPU
- Automatic fallback to CPU if GPU unavailable

### Custom Configuration

You can create a custom YAML configuration file and use it:

```bash
python scripts/train.py --config path/to/config.yaml
```

---

## Implementation Details

### Data Pipeline

**Video Processing:**
1. Videos are loaded using OpenCV
2. Clips of 16 consecutive frames are extracted using a sliding window
3. Frames are resized to 112x112 pixels
4. Clips are normalized using ImageNet statistics

**Temporal Sampling:**
- Training: Overlapping clips with stride=8 frames
- Validation/Testing: Non-overlapping clips with stride=16 frames

**Class Imbalance Handling:**
1. Weighted Random Sampling: Minority classes are oversampled during training
2. Focal Loss: Focuses learning on hard-to-classify examples
3. Class Weights: Loss is weighted by inverse class frequency

### Training Process

**Training Loop:**
1. Forward pass through model
2. Compute Focal Loss with class weights
3. Backward pass with gradient computation
4. Gradient clipping (max norm = 1.0)
5. Optimizer step (AdamW)
6. Metrics tracking (loss, accuracy)

**Validation Loop:**
1. No gradient computation
2. Forward pass only
3. Compute validation loss and accuracy
4. Per-class accuracy tracking

**Checkpointing Strategy:**
- Save best model by validation loss
- Save best model by validation accuracy
- Save periodic checkpoints every 10 epochs
- Each checkpoint includes:
  - Model state dict
  - Optimizer state dict
  - Epoch number
  - Metrics
  - Configuration

**Early Stopping:**
- Monitors validation accuracy
- Stops if no improvement for 10 consecutive epochs

### GradCAM Implementation

**GradCAM Process:**
1. Forward pass through model with target class
2. Extract activations from target layer (default: block5, last conv layer)
3. Compute gradients of target class with respect to activations
4. Weight activations by gradients
5. Generate heatmap by averaging across channels
6. Resize heatmap to input spatial dimensions
7. Overlay on original frames using colormap

**Visualization:**
- Heatmaps are generated for each frame in the clip
- Multiple output formats: original, heatmap, overlay, side-by-side
- Saved as MP4 videos for temporal analysis

### Evaluation Metrics

**Overall Metrics:**
- Accuracy: Percentage of correct predictions
- Top-3 Accuracy: Percentage where true class is in top 3 predictions
- Mean Class Accuracy: Average per-class accuracy (accounts for imbalance)

**Per-Class Metrics:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: Harmonic mean of precision and recall
- Support: Number of samples per class

**Confusion Matrix:**
- Generated in both count and normalized forms
- Visualized as heatmap
- Saved as CSV and PNG

---

## Expected Results

### Performance Targets

Based on the dataset characteristics and model architecture, expected performance metrics are:

**Overall Performance:**
- Overall Accuracy: 70% or higher
- Mean Class Accuracy: 60% or higher (accounts for class imbalance)
- Top-3 Accuracy: 85% or higher

**Per-Class Performance:**
- Precision: 50% or higher for all classes
- Recall: 50% or higher for all classes
- F1-Score: 50% or higher for all classes

### Training Time Estimates

Approximate training times per epoch (with batch size 8):

- NVIDIA GPU (CUDA): 1-2 minutes per epoch
- Apple Silicon (MPS): 2-3 minutes per epoch
- CPU: 15-20 minutes per epoch (not recommended)

Full training (50 epochs):
- GPU: 1-2 hours
- MPS: 2-3 hours
- CPU: 12-16 hours

### Known Limitations

1. **Class Imbalance**: Despite mitigation strategies, the dominant class (Class 1) may still achieve higher accuracy than minority classes

2. **Temporal Context**: 16-frame clips may not capture complete exercise movements for some exercise types

3. **Spatial Resolution**: 112x112 resolution is relatively low and may miss fine-grained details

4. **Device Compatibility**: MPS (Apple Silicon) support may have limitations for some operations compared to CUDA

---

## Troubleshooting

### Common Issues

**Out of Memory (OOM) Errors:**
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Reduce spatial resolution in config
- Reduce number of data loader workers

**Slow Data Loading:**
- Reduce number of workers if experiencing bottlenecks
- Ensure dataset is on fast storage (SSD preferred)

**Training Not Converging:**
- Check learning rate (may need adjustment)
- Verify data augmentation is not too aggressive
- Monitor gradient norms for exploding/vanishing gradients

**GradCAM Errors on MPS:**
- Try running GradCAM on CPU: `--device cpu`
- Some operations in pytorch-grad-cam may not be fully MPS-compatible

---

## References

**Model Architecture:**
- C3D: Learning Spatiotemporal Features with 3D Convolutional Networks (Tran et al., 2015)

**Loss Function:**
- Focal Loss for Dense Object Detection (Lin et al., 2017)

**Visualization:**
- Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (Selvaraju et al., 2017)

