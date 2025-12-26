# MTL Clinical Prediction Model - Final Report

**Date:** December 26, 2025  
**Model Version:** v2.0 (Refactored)

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
│   512 → 256 → 256           │
│   LeakyReLU + BatchNorm     │
│   Dropout 0.2               │
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

**Total Parameters:** 217,413

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 20 |
| Batch Size | 128 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| Scheduler | CosineAnnealingWarmRestarts |
| Weight Decay | 1e-5 |
| Task Balancing | Uncertainty Weighting (Kendall et al., 2018) |

### Class Imbalance Handling

| Task | Strategy | Weights |
|------|----------|---------|
| CVD | Focal Loss (γ=2.0) | - |
| Metabolic | Per-component pos_weight | [None, 3.76, 2.40, 1.78, None] |
| Kidney | Ordinal BCE with pos_weight | [4.5 (Node A), 30.0 (Node B)] |
| Liver | Focal Loss (γ=2.0) | - |

---

## Evaluation Results

### Task A: Cardiovascular Disease

| Metric | @ Threshold 0.5 | @ Optimal 0.33 |
|--------|-----------------|----------------|
| **Recall** | 21% | **80%** |
| **Precision** | 52% | 27% |
| **F1** | 0.30 | 0.40 |
| **ROC-AUC** | 0.83 | - |
| **PR-AUC** | 0.39 | - |

**Production Threshold: 0.33** (for 80% recall)

---

### Task B: Metabolic Syndrome (5 Components)

| Component | Accuracy | Macro-F1 | ROC-AUC |
|-----------|----------|----------|---------|
| Waist | 91% | 0.91 | 0.97 |
| Triglycerides | 66% | 0.63 | 0.74 |
| HDL | 68% | 0.67 | 0.74 |
| Blood Pressure | 67% | 0.67 | 0.75 |
| Glucose | 65% | 0.60 | 0.68 |

**Overall Micro-F1: 0.73**

---

### Task C: Kidney Dysfunction (Ordinal)

| Class | Recall | Count |
|-------|--------|-------|
| Normal (ACR <30) | 82% | 5,681 |
| Microalbuminuria (30-300) | 22% | 1,003 |
| Macroalbuminuria (>300) | **51%** | 203 |

**Ordinal Encoding:** [0,0] → Normal, [1,0] → Micro, [1,1] → Macro

**Key Achievement:** Model detects 51% of Macroalbuminuria cases (rare, clinically critical class).

---

### Task D: Liver Dysfunction

| Metric | @ Threshold 0.5 | @ Optimal 0.44 |
|--------|-----------------|----------------|
| **Recall** | 60% | **70%** |
| **Precision** | 83% | 77% |
| **F1** | 0.70 | 0.73 |
| **ROC-AUC** | 0.93 | - |

**Production Threshold: 0.44** (for 70% recall)

---

## Production Configuration

```python
# Optimal thresholds (from 06_threshold_optimization.py)
OPTIMAL_THRESHOLDS = {
    'cvd': 0.33,      # For 80% recall
    'liver': 0.44,    # For 70% recall
}

# Kidney ordinal weights
ORDINAL_WEIGHTS = [4.5, 30.0]  # [ACR≥30, ACR≥300]
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
| `06_threshold_optimization.py` | Find optimal decision thresholds |
| `07_streamlit_app.py` | Web interface for predictions |
| `trained_model.pth` | PyTorch checkpoint |
| `trained_model.onnx` | ONNX format for deployment |

---

## Key Innovations

1. **Uncertainty Weighting** - Automatic task balancing via learned variance
2. **Ordinal Binary Decomposition** - Enforces disease severity progression
3. **Threshold Optimization** - Clinical calibration for target recall
4. **Focal Loss** - Focuses on hard examples for imbalanced classes
5. **Gradient Clipping** - Stability with aggressive minority class weights

---

## Recommendations

1. **Deploy with optimized thresholds** - Use 0.33 for CVD, 0.44 for Liver
2. **Monitor Node A/B accuracy** - Track ordinal decomposition performance
3. **Consider ensemble** - Combine with simpler models for robustness
4. **Recalibrate quarterly** - Population drift may shift optimal thresholds
