# EDA Summary: Actionable Findings for Model Training

**Generated from:** 7 EDA notebooks (executed 2025-12-26)  
**Dataset:** NHANES Multi-Task Learning Dataset  
**Post-Harmonization Size:** 34,097 adults (Age ≥ 20)

---

## Executive Summary

| Finding | Impact | Required Action |
|---------|--------|-----------------|
| **3 columns >50% missing** | Target signal dilution | Use masked loss for these targets |
| **CVD class imbalance (88:12)** | Model bias toward healthy | Implement `pos_weight` in BCELoss |
| **Albuminuria Class 2 (2.0%)** | Rare class ignored | Add class weights to CrossEntropyLoss |
| **11,679 duplicate rows** | Expected (adults filter) | Already handled in pipeline |
| **8 columns with impossible values** | Already clamped | No further action needed |

---

## 1. Dataset Overview (Fresh Results)

| Metric | Value |
|--------|-------|
| **Total Records** | 34,097 |
| **Total Features** | 29 (21 inputs + 8 targets) |
| **Age Range** | 20 - 80 years |
| **Mean Age** | 51.2 years |
| **Mean BMI** | 29.7 kg/m² |

---

## 2. Missing Values Analysis

### Columns with >50% Missing (Targets)

| Column | Missing | Percent | Impact |
|--------|---------|---------|--------|
| `high_glucose_mg_dl` | 20,281 | **59.5%** | Fasting-only subset |
| `high_triglycerides_mg_dl` | 19,884 | **58.3%** | Fasting-only subset |
| `smoking_status` | 19,761 | **58.0%** | Adult questionnaire gaps |

**Training Action:** These targets have majority missing. Use masked loss with lower task weights.

### Moderate Missing (10-30%)

| Column | Missing | Percent |
|--------|---------|---------|
| `alcohol_drinks_per_week` | 10,509 | 30.8% |
| `heart_rate_bpm` | 6,622 | 19.4% |
| `liver_ast_U_L` | 5,445 | 16.0% |
| `liver_dysfunction` | 5,389 | 15.8% |
| `albuminuria_risk` | 3,926 | 11.5% |

**Training Action:** IterativeImputer (MICE) handles input missingness. Target NaNs → masked loss.

---

## 3. Target Variable Distributions

### Task A: Cardiovascular Disease (Binary)

| Class | Count | Percentage |
|-------|-------|------------|
| Healthy (0) | 30,018 | **88.0%** |
| Has CVD (1) | 4,079 | **12.0%** |

> ⚠️ **Class Imbalance Ratio: 7.4:1**

**Training Actions:**
```python
# Option 1: Weighted BCELoss
pos_weight = torch.tensor([30018 / 4079])  # = 7.36
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Option 2: Focal Loss (for hard examples)
# alpha=0.25, gamma=2.0 (standard parameters)
```

---

### Task B: Metabolic Syndrome (5 Binary Components)

| Component | Positive (1) | Negative (0) | Missing | Imbalance |
|-----------|--------------|--------------|---------|-----------|
| `high_waist_circumference` | 17,330 (50.8%) | 11,810 (34.6%) | 4,957 (14.5%) | ✅ Balanced |
| `high_blood_pressure` | 10,328 (30.3%) | 18,370 (53.9%) | 5,399 (15.8%) | 🟡 Moderate |
| `low_hdl_mg_dl` | 8,485 (24.9%) | 20,370 (59.7%) | 5,242 (15.4%) | 🟡 Moderate |
| `high_triglycerides_mg_dl` | 2,985 (8.8%) | 11,228 (32.9%) | **19,884 (58.3%)** | ⚠️ High missing |
| `high_glucose_mg_dl` | 8,129 (23.8%) | 5,687 (16.7%) | **20,281 (59.5%)** | ⚠️ High missing |

**Training Actions:**
1. Apply masked loss per-component (already implemented)
2. Consider lower task weights for glucose/triglycerides:
   ```python
   metabolic_weights = {'waist': 1.0, 'bp': 1.0, 'hdl': 1.0, 'trig': 0.5, 'glucose': 0.5}
   ```
3. Report per-component metrics, not just aggregate

---

### Task C: Albuminuria Risk (3-Class Ordinal)

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| 0 | Normal (ACR <30) | 26,217 | **76.9%** |
| 1 | Microalbuminuria (30-300) | 3,265 | **9.6%** |
| 2 | Macroalbuminuria (>300) | 689 | **2.0%** |
| NaN | Missing | 3,926 | 11.5% |

> ⚠️ **Extreme Imbalance: 38:5:1 ratio**

**Training Actions:**
```python
# Class weights inversely proportional to frequency
class_counts = torch.tensor([26217, 3265, 689], dtype=torch.float)
weights = 1.0 / class_counts
weights = weights / weights.sum() * 3  # Normalize to sum=3

criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
# -1 for masked samples (convert NaN to -1 before training)
```

> **Critical:** Always report per-class F1, especially for Class 2 (Macro)

---

### Task D: Liver Dysfunction (Binary, Gender-Adjusted)

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (0) | 24,373 | **71.5%** |
| Dysfunction (1) | 4,335 | **12.7%** |
| Missing | 5,389 | 15.8% |

> **Imbalance Ratio: 5.6:1**

**Training Action:** Use `pos_weight = 5.6` in BCEWithLogitsLoss

---

## 4. Data Quality (Post-Harmonization)

### Impossible Values Detected

| Column | Count | Resolution |
|--------|-------|------------|
| `liver_ggt_U_L` | 66 | ✅ Outside range [5, 500] - still in data |
| `bilirubin_mg_dl` | 29 | ✅ Outside range [0.1, 20] - clamped |
| `creatinine_mg_dl` | 6 | ✅ Outside range [0.3, 15] |
| `uric_acid_mg_dl` | 6 | ✅ Outside range [1, 15] |
| `potassium_mmol_L` | 6 | ✅ Outside range [2.5, 6.0] |
| `cholesterol_mg_dl` | 5 | ✅ Outside range [50, 500] |
| `sodium_mmol_L` | 3 | ✅ Outside range [120, 160] |
| `liver_ast_U_L` | 2 | ✅ Outside range [5, 500] |

**Note:** These are edge cases outside EDA ranges but may be clinically valid. Harmonization pipeline already clamps extreme biological impossibilities.

### Duplicate Rows

| Metric | Count |
|--------|-------|
| Duplicate rows | 11,679 (34.3%) |
| Unique rows | 22,418 |

**Note:** Some duplicates expected due to categorical combinations (e.g., same age, gender, ethnicity patterns). No action required - model training uses all rows.

---

## 5. Key Statistics for Model Config

### Input Feature Statistics

| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| `age` | 51.2 | 17.7 | 20.0 | 80.0 |
| `body_mass_index` | 29.7 | 7.3 | 11.1 | 92.3 |
| `heart_rate_bpm` | 71.3 | 11.9 | 34.0 | 160.0 |
| `hemoglobin_g_dl` | ~14.0 | ~1.5 | - | - |

### Recommended Hyperparameters

Based on EDA findings:

```python
# Model Configuration
config = {
    # Class weights for imbalanced targets
    'cvd_pos_weight': 7.36,
    'liver_pos_weight': 5.62,
    'kidney_class_weights': [0.073, 0.59, 2.8],  # Inverse frequency
    
    # Task weights (reduce for high-missing targets)
    'task_weights': {
        'cvd': 1.0,
        'metabolic_waist': 1.0,
        'metabolic_bp': 1.0,
        'metabolic_hdl': 1.0,
        'metabolic_trig': 0.5,  # 58% missing
        'metabolic_glucose': 0.5,  # 60% missing
        'albuminuria': 1.0,
        'liver': 1.0
    },
    
    # Regularization (for multicollinearity)
    'weight_decay': 1e-4,
    
    # Learning rate
    'lr': 1e-3
}
```

---

## 6. Warnings for Model Training

> [!CAUTION]
> **Albuminuria Class 2 (Macroalbuminuria) has only 689 samples (2.0%)**
> Without class weighting, model will achieve 98% accuracy by predicting Class 0 always.

> [!WARNING]
> **Glucose and Triglyceride targets have ~60% missing**
> Consider separate evaluation on fasting-only subset for these components.

> [!IMPORTANT]
> **Masked loss is CRITICAL**
> Never fill target NaNs with zeros - this creates false "healthy" labels.

---

## 7. Summary Table

| Task | Type | Classes | Imbalance | Missing % | Weight Strategy |
|------|------|---------|-----------|-----------|-----------------|
| CVD | Binary | 2 | 7.4:1 | 0% | `pos_weight=7.36` |
| Waist | Binary | 2 | ~1:1 | 14.5% | None needed |
| BP | Binary | 2 | ~2:1 | 15.8% | Mild weighting |
| HDL | Binary | 2 | ~2.4:1 | 15.4% | Mild weighting |
| Triglycerides | Binary | 2 | ~3.8:1 | **58.3%** | Weight + task weight |
| Glucose | Binary | 2 | ~1.4:1 | **59.5%** | Task weight only |
| Albuminuria | 3-Class | 3 | 38:5:1 | 11.5% | Class weights |
| Liver | Binary | 2 | 5.6:1 | 15.8% | `pos_weight=5.62` |

---

**End of EDA Summary**

*Generated: December 26, 2025*
