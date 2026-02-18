

https://github.com/user-attachments/assets/67d3b5ef-41c2-4e16-9f8b-961cad79f947



# SXCROL – Offroad Semantic Scene Segmentation

Hackathon Submission to Duality AI

---

# 1. Introduction

This repository contains the complete implementation of our semantic segmentation pipeline developed for the Duality AI Offroad Scene Segmentation Hackathon.

The objective of this challenge was to build a high-performance pixel-wise classification model capable of accurately segmenting complex off-road terrain scenes into predefined semantic classes while maintaining real-time inference speed.

Our final model achieves:

**Mean IoU:** 0.64
**Inference Speed:** ~38 ms per image
**Training Stability:** Smooth convergence with no major overfitting

---

# 2. Problem Statement

Off-road scene understanding is challenging due to:

* High inter-class similarity (logs vs dry bushes)
* Severe class imbalance (flowers and logs are rare)
* Texture ambiguity (rocks vs ground clutter)
* Lighting variations and shadows
* Small object segmentation difficulty

The task required accurate pixel-level classification across 10 terrain/object categories while ensuring inference time remains under 50 ms.

---

# 3. Classes Segmented

The model predicts the following 10 classes:

1. Sky
2. Landscape
3. Trees
4. Rocks
5. Lush Bushes
6. Dry Bushes
7. Dry Grass
8. Ground Clutter
9. Flowers
10. Logs

---

# 4. Our Approach

## 4.1 Model Architecture

We selected:

DeepLabV3+ with ResNet-50 backbone (ImageNet pretrained)

### Why DeepLabV3+?

* ASPP module captures multi-scale context
* Strong boundary refinement
* Efficient tradeoff between speed and accuracy
* Proven benchmark performance in segmentation tasks

---

## 4.2 Data Preprocessing

* All images resized to 512 × 512
* Normalized using ImageNet statistics
* Masks converted to integer class indices
* Ensured consistent data loading pipeline

---

## 4.3 Data Augmentation Strategy

To improve generalization and reduce overfitting, we applied:

* Random Horizontal Flip
* Random Rotation
* Color Jitter (brightness, contrast)
* Random Resized Crop
* Minor perspective perturbations

Impact:

* Improved minority class detection
* Reduced confusion between similar textures
* Enhanced robustness to lighting variation

---

# 5. Training Strategy

## Hyperparameters

Epochs: 35
Batch Size: 8
Optimizer: AdamW
Learning Rate: 0.0005
Weight Decay: 0.0001
Scheduler: Cosine Annealing
Dropout: 0.3
Mixed Precision: Enabled
Random Seed: 42

---

## 5.1 Loss Function Design

We used a hybrid loss:

Weighted Cross Entropy + Dice Loss

### Why?

Cross Entropy:

* Ensures accurate pixel classification

Dice Loss:

* Maximizes overlap between predicted and ground truth masks
* Helps with small object segmentation

Weighted CE:

* Addresses severe class imbalance

This combination significantly boosted minority class IoU.

---

# 6. Performance Results

## Baseline Model

Configuration:

* No augmentation
* Standard Cross Entropy
* Basic optimizer setup

Mean IoU: 0.31

---

## Final Optimized Model

Mean IoU: **0.64**

Absolute Improvement: +0.33
Relative Improvement: +106%

---

## Per-Class IoU

Sky: 0.91
Landscape: 0.87
Trees: 0.77
Rocks: 0.71
Lush Bushes: 0.68
Dry Bushes: 0.63
Dry Grass: 0.59
Ground Clutter: 0.52
Flowers: 0.48
Logs: 0.50

Observations:

* Large structured classes achieved high accuracy.
* Small and texture-similar objects remain challenging.
* Significant improvement observed in rare classes due to weighted loss and augmentation.

---

# 7. Training Curve Behavior

Training Loss:

* Smooth decrease from 1.85 → 0.32

Validation Loss:

* Stabilized around 0.40

Mean IoU:

* Steady improvement after epoch 5
* Plateau around epoch 32

Cosine annealing scheduler helped smooth late-stage convergence.

No severe overfitting observed.

---

# 8. Failure Case Analysis

## 8.1 Logs vs Dry Bushes

Issue:
Texture and color similarity.

Mitigation:

* Dice loss improved boundary learning
* Multi-scale augmentation enhanced contextual understanding

Remaining Limitation:
Partial occlusion still causes misclassification.

---

## 8.2 Rocks vs Ground Clutter

Issue:
Visual ambiguity and similar surface patterns.

Mitigation:
Weighted loss improved separation but fine-grained texture remains challenging.

---

## 8.3 Small Flowers in Distance

Issue:
Very small pixel representation.

Mitigation:
Random resized crop increased effective object resolution.

---

# 9. Inference Performance

Average Inference Time: ~38 ms per image
Hardware Tested: NVIDIA T4 GPU

The model satisfies real-time constraint (<50 ms).

Memory footprint remains manageable for deployment scenarios.

---

# 10. Reproducibility

To ensure reproducibility:

* Random seed fixed to 42
* Deterministic data loading
* No test data used during training
* Config file included
* Full training & inference scripts provided

---

# 11. How to Reproduce Results

## 11.1 Install Dependencies

```
pip install -r requirements.txt
```

---

## 11.2 Dataset Structure

```
dataset/
│
├── Train/
│   ├── RGB/
│   └── Masks/
│
├── Val/
│   ├── RGB/
│   └── Masks/
│
testImages/
│   └── RGB/
```

---

## 11.3 Train the Model

```
python train.py \
  --data_dir ./dataset \
  --config config.yaml \
  --save_dir ./runs/
```

Best model saved as:

```
./runs/best_model.pth
```

---

## 11.4 Run Inference

```
python test.py \
  --model_path ./runs/best_model.pth \
  --test_dir ./testImages \
  --output_dir ./predictions/
```

Predictions will be saved in `./predictions/`.

---

# 12. Repository Structure

```
SXCROL_DualityAI_Submission/
│
├── train.py
├── test.py
├── config.yaml
├── model_weights.pth
├── requirements.txt
├── README.md
├── SXCROL_Hackathon_Report.txt
│
├── utils/
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── augmentations.py
│   └── visualise.py
│
└── predictions/
```

---

# 13. Team Contributions

Varun (Team Lead)

* Model architecture selection
* Training pipeline design
* Optimization strategy

Vrinda

* Data preprocessing
* Augmentation engineering

Vanya

* Metric computation
* IoU validation

Smridhi

* Documentation
* Failure analysis

---

# 14. Conclusion

Team SXCROL successfully developed a robust semantic segmentation system for off-road scene understanding.

Key Achievements:

* Mean IoU of 0.64
* 38 ms inference speed
* Significant improvement over baseline
* Balanced performance across large and small classes
* Fully reproducible pipeline

The system demonstrates strong potential for real-time terrain perception applications.

---

# 15. Future Work

* Upgrade backbone to ResNet-101
* Integrate Focal Loss
* Multi-scale test-time augmentation
* Lightweight backbone for edge deployment

---

# Submission Details

* Private GitHub repository submitted
* All required scripts, weights, configs included
* Reproducibility instructions provided

---

Team SXCROL


---

