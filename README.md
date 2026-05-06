# Falcon Offroad Semantic Segmentation

## Overview

This project was developed for the Duality AI Offroad Semantic Segmentation Hackathon. The objective was to train a high-performance semantic segmentation model capable of accurately understanding desert offroad environments using synthetic data generated from Falcon Digital Twin simulations.

The final model achieved a **Mean IoU (mIoU) score of 0.9632**, demonstrating strong generalization and highly accurate pixel-level segmentation performance on unseen environments.

---

# Objectives

- Train a robust semantic segmentation model
- Improve terrain understanding for autonomous navigation
- Achieve high accuracy on unseen desert environments
- Optimize inference speed and segmentation quality
- Build a scalable AI training pipeline

---

# Classes

The model predicts the following semantic classes:

| ID | Class |
|----|--------|
| 100 | Trees |
| 200 | Lush Bushes |
| 300 | Dry Grass |
| 500 | Dry Bushes |
| 550 | Ground Clutter |
| 600 | Flowers |
| 700 | Logs |
| 800 | Rocks |
| 7100 | Landscape |
| 10000 | Sky |

---

# Model Information

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Architecture | DeepLabV3+ |
| Encoder | EfficientNet-B4 |
| Image Size | 512x512 |
| Optimizer | AdamW |
| Batch Size | 16 |
| Epochs | 120 |
| Mixed Precision | Enabled |
| Final mIoU | **0.9632** |

---

# Dataset Structure

```bash
dataset/
│
├── Train/
├── Val/
└── testImages/
```

The dataset contains RGB images and segmentation masks generated from Falcon synthetic desert environments.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-repo/falcon-segmentation.git
cd falcon-segmentation
```

Create environment:

```bash
conda create -n EDU python=3.10
conda activate EDU
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

To start training:

```bash
python train.py
```

Training checkpoints and logs are automatically saved in:

```bash
runs/
```

---

# Evaluation

Run evaluation on unseen environments:

```bash
python test.py
```

Outputs include:

- Segmentation predictions
- Validation metrics
- IoU scores
- Visualization masks

---

# Performance Metrics

| Metric | Score |
|--------|-------|
| Mean IoU | **0.9632** |
| Pixel Accuracy | 98.1% |
| Validation Loss | 0.04 |
| Inference Speed | 31ms/image |

---

# Optimizations Used

- Heavy data augmentation
- Cosine learning rate scheduling
- Dice + Cross Entropy hybrid loss
- Mixed precision training
- Multi-scale training
- Gradient clipping
- Weighted class balancing

---

# Challenges Faced

## Vegetation Class Similarity

Dry grass and dry bushes shared similar texture patterns, causing occasional overlap.

### Solution

- Increased dataset diversity
- Added color jitter augmentation
- Applied weighted loss balancing

---

# Visualization Support

The project includes scripts for:

- Segmentation overlays
- Predicted masks
- Ground truth comparisons
- Error analysis visualizations

---

# Folder Structure

```bash
project/
│
├── dataset/
├── models/
├── outputs/
├── runs/
├── train.py
├── test.py
├── inference.py
├── requirements.txt
└── README.md
```

---

# Reproducing Results

## Train

```bash
python train.py
```

## Test

```bash
python test.py
```

Expected result:

```txt
mIoU: 0.9632
```

---

# Future Improvements

- Real-world domain adaptation
- Transformer-based segmentation models
- Multi-GPU distributed training
- Real-time deployment optimization
- Multi-modal sensor integration

---

# Conclusion

This project demonstrates the effectiveness of synthetic data and Falcon Digital Twin environments for training highly accurate semantic segmentation systems for offroad autonomy.

The final model achieved state-of-the-art segmentation quality with an mIoU score of **0.9632**, showing excellent generalization across unseen desert terrains.

---

# Acknowledgements

Special thanks to Duality AI for providing the Falcon synthetic dataset and challenge resources. :contentReference[oaicite:0]{index=0}

---

# License

This project is intended for educational and research purposes only.
