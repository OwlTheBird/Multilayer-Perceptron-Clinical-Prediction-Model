# MTL Clinical Prediction Model - Final Report

**Date:** December 27, 2025  
**Model Version:** v3.0 (Hyperparameter Optimized)

---

## Executive Summary

This Multi-Task Learning (MTL) model predicts 4 clinical outcomes from NHANES biomarker data:
- **Cardiovascular Disease (CVD)** - Binary classification
- **Metabolic Syndrome** - 5-component multi-label classification
- **Kidney Dysfunction** - 3-class ordinal classification (via 2-node binary decomposition)
- **Liver Dysfunction** - Binary classification

---

## Model Architecture

```
Input: 30 continuous biomarkers
         │
         ▼
┌─────────────────────────────┐
│   Shared Backbone           │
│   1024 → 768 → 512          │
│   LeakyReLU + BatchNorm     │
│   Dropout 0.05              │
└─────────────────────────────┘
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ CVD  │ │ Met  │ │Kidney│ │Liver │
│ (1)  │ │ (5)  │ │ (2)  │ │ (1)  │
└──────┘ └──────┘ └──────┘ └──────┘
  Focal    BCE    Ordinal   Focal
  Loss     Loss    BCE      Loss
```

**Total Parameters:** ~500K (optimized architecture)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch Size | 128 |
| Learning Rate | 2.4e-4 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingWarmRestarts |
| Weight Decay | 4.2e-4 |
| Dropout | 0.05 |
| Task Balancing | Uncertainty Weighting (Kendall et al., 2018) |

### Class Imbalance Handling

| Task | Strategy | Weights |
|------|----------|---------|
| CVD | Focal Loss (γ=1.0) | - |
| Metabolic | Per-component pos_weight | [None, 3.76, 2.40, 1.78, None] |
| Kidney | Ordinal BCE with pos_weight | [4.5 (Node A), 5.0 (Node B)] |
| Liver | Focal Loss (γ=1.0) | - |

---

## Evaluation Results

### Task A: Cardiovascular Disease

| Metric | Value |
|--------|-------|
| **Accuracy** | 83.5% |
| **Precision** | 36.1% |
| **Recall** | 55.8% |
| **F1 Score** | 0.4382 |
| **Macro-F1** | **0.6706** |
| **ROC-AUC** | **0.8331** |
| **PR-AUC** | 0.4004 |

**Confusion Matrix:**
```
              Pred: 0    Pred: 1
True: 0        5649        838
True: 1         375        473
```

---

### Task B: Metabolic Syndrome (5 Components)

| Component | Accuracy | Macro-F1 | ROC-AUC | PR-AUC |
|-----------|----------|----------|---------|--------|
| Waist | **90.9%** | **0.9040** | **0.9711** | 0.9818 |
| Triglycerides | 67.4% | 0.6343 | 0.7323 | 0.4931 |
| HDL | 67.5% | 0.6592 | 0.7375 | 0.5858 |
| Blood Pressure | 67.0% | 0.6667 | 0.7369 | 0.6191 |
| Glucose | 65.4% | 0.5957 | 0.6865 | 0.7649 |

**Overall Micro-F1: 0.7247**

---

### Task C: Kidney Dysfunction (Ordinal)

| Metric | Value |
|--------|-------|
| **Accuracy** | 75.5% |
| **Macro-Precision** | 50.3% |
| **Macro-Recall** | 50.4% |
| **Macro-F1** | **0.5008** |
| **Weighted-F1** | 0.7656 |

**Per-Class Recall:**

| Class | Recall | Count |
|-------|--------|-------|
| Normal (ACR <30) | **84.0%** | 5,681 |
| Microalbuminuria (30-300) | **37.1%** | 1,003 |
| Macroalbuminuria (>300) | 30.1% | 203 |

**Ordinal Encoding:** [0,0] → Normal, [1,0] → Micro, [1,1] → Macro

---

### Task D: Liver Dysfunction

| Metric | Value |
|--------|-------|
| **Accuracy** | **88.3%** |
| **Precision** | 77.8% |
| **Recall** | 66.1% |
| **F1 Score** | 0.7147 |
| **Macro-F1** | **0.8204** |
| **ROC-AUC** | **0.9217** |
| **PR-AUC** | 0.8094 |

**Confusion Matrix:**
```
              Pred: 0    Pred: 1
True: 0        4884        279
True: 1         501        977
```

---

## Production Configuration

```python
# Optimized hyperparameters (from Optuna tuning)
HIDDEN_DIM = 512
DROPOUT_RATE = 0.05
LEARNING_RATE = 0.00024
BATCH_SIZE = 128
WEIGHT_DECAY = 0.00042
FOCAL_GAMMA = 1.0

# Kidney ordinal weights (tuned)
KIDNEY_ORDINAL_WEIGHTS = [4.5, 5.0]

# Optimal thresholds
OPTIMAL_THRESHOLDS = {
    'cvd': 0.33,      # For higher recall
    'liver': 0.44,    # For 70% recall
}
```

---

## Files in Model Directory

| File | Description |
|------|-------------|
| `01_config.py` | Configuration and hyperparameters |
| `02_dataset.py` | Data loading with ordinal encoding |
| `03_model.py` | SharedBottomMTL architecture |
| `04_train.py` | Training loop with uncertainty weighting |
| `05_evaluate.py` | Evaluation metrics and ordinal decoding |
| `06_hyperparameter_tuning.py` | Optuna-based HPO |
| `07_streamlit_app.py` | Web interface for predictions |
| `trained_model.pth` | PyTorch checkpoint |
| `trained_model.onnx` | ONNX format for deployment |
| `best_params.json` | Best hyperparameters from tuning |

---

## Key Innovations

1. **Hyperparameter Optimization** - Optuna with Harmonic Mean objective
2. **Uncertainty Weighting** - Automatic task balancing via learned variance
3. **Ordinal Binary Decomposition** - Enforces disease severity progression
4. **Threshold Optimization** - Clinical calibration for target recall
5. **Focal Loss** - Focuses on hard examples for imbalanced classes
6. **Gradient Clipping** - Stability with aggressive minority class weights

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v3.0 | 2025-12-27 | Hyperparameter optimization, reduced kidney weight |
| v2.0 | 2025-12-26 | Refactored architecture |
| v1.0 | 2025-12-20 | Initial model |
