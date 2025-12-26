# EDA Summary: Comprehensive Column Analysis

**Generated from:** 7 EDA notebooks (01-07)  
**Dataset:** NHANES Multi-Task Learning Dataset  
**Post-Harmonization Size:** 34,097 adults (Age ≥ 20)  
**Total Features:** 29 (21 input features + 8 target variables)  
**Last Updated:** December 26, 2025

---

## Executive Summary

| Finding | Impact | Required Action |
|---------|--------|-----------------|
| **3 columns >50% missing** | Target signal dilution | Use masked loss for these targets |
| **CVD class imbalance (88:12)** | Model bias toward healthy | Implement `pos_weight` in BCELoss |
| **Albuminuria Class 2 (2.0%)** | Rare class ignored | Add class weights to CrossEntropyLoss |
| **5,940 duplicate rows** | Expected (pattern combinations) | Already handled in pipeline |
| **8 columns with outliers >5%** | Potential noise | Use robust scaling methods |

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Records** | 34,097 |
| **Total Features** | 29 |
| **Input Features** | 21 |
| **Target Variables** | 8 |
| **Age Range** | 20 - 80 years |
| **Mean Age** | 51.2 years |
| **Mean BMI** | 29.7 kg/m² |
| **Memory Usage** | 7.54 MB |

---

## Legend

> 🎯 **TARGET COLUMN** - This is a prediction target for the multi-task learning model

---

# Complete Column Reference

## 1. Demographic Features

### `age`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 34,097 (100.0%) |
| **Missing** | 0 (0.0%) |
| **Mean ± Std** | 51.16 ± 17.73 |
| **Range** | 20.0 - 80.0 |
| **Median [Q1, Q3]** | 52.0 [36.0, 66.0] |
| **Skewness / Kurtosis** | -0.03 / -1.16 |

**Clinical Note:** Adult population (≥20) with slight left skew indicating slightly older population center.

---

### `gender`
| Attribute | Value |
|-----------|-------|
| **Type** | Categorical (Binary) |
| **Non-Null** | 34,097 (100.0%) |
| **Missing** | 0 (0.0%) |
| **Categories** | 1 = Male, 2 = Female |

| Category | Count | Percentage |
|----------|-------|------------|
| Male (1) | 16,173 | 47.4% |
| Female (2) | 17,924 | 52.6% |

**Clinical Note:** Slightly more female participants, reflective of NHANES sampling.

---

### `ethnicity`
| Attribute | Value |
|-----------|-------|
| **Type** | Categorical |
| **Non-Null** | 34,097 (100.0%) |
| **Missing** | 0 (0.0%) |
| **Categories** | 6 unique values |

| Category Code | Count | Percentage |
|---------------|-------|------------|
| 3 (Non-Hispanic White) | 14,042 | 41.2% |
| 4 (Non-Hispanic Black) | 7,127 | 20.9% |
| 1 (Mexican American) | 4,099 | 12.0% |
| 6 (Other Hispanic) | 3,710 | 10.9% |
| 2 (Other Race) | 3,512 | 10.3% |
| 7 (Asian) | 1,607 | 4.7% |

**Clinical Note:** NHANES oversamples minority populations for statistical precision.

---

### `income_ratio`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 29,491 (86.5%) |
| **Missing** | 4,606 (13.5%) |
| **Mean ± Std** | 2.61 ± 1.64 |
| **Range** | 0.0 - 5.0 |
| **Median [Q1, Q3]** | 2.25 [1.18, 4.27] |
| **Skewness / Kurtosis** | 0.26 / -1.35 |

**Clinical Note:** Poverty-income ratio (family income / poverty threshold). Values >5 are top-coded.

---

## 2. Physical Measurements

### `body_mass_index`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 30,452 (89.3%) |
| **Missing** | 3,645 (10.7%) |
| **Mean ± Std** | 29.71 ± 7.34 |
| **Range** | 11.1 - 92.3 |
| **Median [Q1, Q3]** | 28.5 [24.6, 33.4] |
| **Skewness / Kurtosis** | 1.28 / 3.18 |
| **IQR Outliers** | 906 (3.0%) |

**Clinical Note:** Mean BMI in overweight range (25-30). Right-skewed with extreme obesity outliers.

---

### `height_cm`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 30,524 (89.5%) |
| **Missing** | 3,573 (10.5%) |
| **Mean ± Std** | 166.60 ± 10.12 |
| **Range** | 129.7 - 202.7 |
| **Median [Q1, Q3]** | 166.3 [159.1, 173.9] |
| **Skewness / Kurtosis** | 0.13 / -0.42 |

**Clinical Note:** Approximately normal distribution, gender-differentiated.

---

### `heart_rate_bpm`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 27,475 (80.6%) |
| **Missing** | 6,622 (19.4%) |
| **Mean ± Std** | 71.33 ± 11.88 |
| **Range** | 34.0 - 160.0 |
| **Median [Q1, Q3]** | 70.0 [63.0, 78.0] |
| **Skewness / Kurtosis** | 0.59 / 0.86 |

**Clinical Note:** Normal resting heart rate range (60-100 bpm). Higher missing rate due to measurement protocols.

---

## 3. Blood Tests

### `white_blood_cells_count`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 29,492 (86.5%) |
| **Missing** | 4,605 (13.5%) |
| **Mean ± Std** | 7.21 ± 2.21 (×10³/μL) |
| **Range** | 1.4 - 44.8 |
| **Median [Q1, Q3]** | 6.9 [5.7, 8.4] |
| **Skewness / Kurtosis** | 1.75 / 14.35 |
| **IQR Outliers** | 655 (2.2%) |

**Clinical Note:** Normal range 4.5-11.0 ×10³/μL. High kurtosis indicates some extreme values (infection/inflammation).

---

### `platelets_count`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 29,492 (86.5%) |
| **Missing** | 4,605 (13.5%) |
| **Mean ± Std** | 244.0 ± 64.6 (×10³/μL) |
| **Range** | 36.0 - 787.0 |
| **Median [Q1, Q3]** | 237.0 [200.0, 280.0] |
| **Skewness / Kurtosis** | 0.88 / 2.51 |

**Clinical Note:** Normal range 150-400 ×10³/μL. Moderate right skew.

---

### `hemoglobin_g_dl`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 29,498 (86.5%) |
| **Missing** | 4,599 (13.5%) |
| **Mean ± Std** | 13.94 ± 1.54 |
| **Range** | 5.4 - 19.9 |
| **Median [Q1, Q3]** | 14.0 [13.0, 15.0] |
| **Skewness / Kurtosis** | -0.42 / 1.00 |

**Clinical Note:** Normal range: Men 13.5-17.5, Women 12.0-15.5 g/dL. Slight left skew (anemic cases).

---

### `mean_corpuscular_volume_fL`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 29,498 (86.5%) |
| **Missing** | 4,599 (13.5%) |
| **Mean ± Std** | 88.69 ± 6.06 |
| **Range** | 35.4 - 116.7 |
| **Median [Q1, Q3]** | 89.2 [85.8, 92.4] |
| **Skewness / Kurtosis** | -0.99 / 3.52 |

**Clinical Note:** Normal range 80-100 fL. Correlates with anemia types.

---

## 4. Kidney Function Markers

### `creatinine_mg_dl`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,717 (84.2%) |
| **Missing** | 5,380 (15.8%) |
| **Mean ± Std** | 0.90 ± 0.48 |
| **Range** | 0.25 - 17.41 |
| **Median [Q1, Q3]** | 0.84 [0.71, 1.0] |
| **Skewness / Kurtosis** | 13.68 / 300.34 |

**Clinical Note:** Normal: 0.7-1.3 mg/dL (men), 0.6-1.1 mg/dL (women). Extreme right skew indicates kidney disease cases.

---

### `uric_acid_mg_dl`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,713 (84.2%) |
| **Missing** | 5,384 (15.8%) |
| **Mean ± Std** | 5.38 ± 1.46 |
| **Range** | 0.7 - 18.0 |
| **Median [Q1, Q3]** | 5.3 [4.3, 6.3] |
| **Skewness / Kurtosis** | 0.59 / 0.90 |

**Clinical Note:** Normal: 3.4-7.0 mg/dL (men), 2.4-6.0 mg/dL (women). Associated with gout and kidney stones.

---

## 5. Liver Function Markers

### `liver_ast_U_L`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,652 (84.0%) |
| **Missing** | 5,445 (16.0%) |
| **Mean ± Std** | 23.40 ± 15.47 |
| **Range** | 6.0 - 882.0 |
| **Median [Q1, Q3]** | 21.0 [17.0, 26.0] |
| **Skewness / Kurtosis** | 17.00 / 681.16 |
| **IQR Outliers** | 1,483 (5.2%) |

**Clinical Note:** Normal <40 U/L. Extremely right-skewed with liver disease cases.

---

### `bilirubin_mg_dl`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,693 (84.1%) |
| **Missing** | 5,404 (15.8%) |
| **Mean ± Std** | 0.52 ± 0.32 |
| **Range** | 0.01 - 14.2 |
| **Median [Q1, Q3]** | 0.5 [0.3, 0.6] |
| **Skewness / Kurtosis** | 7.18 / 226.98 |

**Clinical Note:** Normal 0.1-1.2 mg/dL. Elevated in liver/bile duct problems.

---

### `liver_ggt_U_L`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,715 (84.2%) |
| **Missing** | 5,382 (15.8%) |
| **Mean ± Std** | 30.66 ± 48.11 |
| **Range** | 2.0 - 2394.0 |
| **Median [Q1, Q3]** | 20.0 [14.0, 31.0] |
| **Skewness / Kurtosis** | 14.49 / 409.50 |
| **IQR Outliers** | 2,531 (8.8%) |

**Clinical Note:** Normal 9-48 U/L. Sensitive marker for alcohol and bile duct issues.

---

## 6. Electrolytes

### `sodium_mmol_L`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,756 (84.3%) |
| **Missing** | 5,341 (15.7%) |
| **Mean ± Std** | 139.81 ± 2.59 |
| **Range** | 119.0 - 167.0 |
| **Median [Q1, Q3]** | 140.0 [138.0, 141.0] |
| **Skewness / Kurtosis** | -0.23 / 2.71 |

**Clinical Note:** Normal 136-145 mmol/L. Tight physiological regulation.

---

### `potassium_mmol_L`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,673 (84.1%) |
| **Missing** | 5,424 (15.9%) |
| **Mean ± Std** | 4.05 ± 0.36 |
| **Range** | 2.5 - 7.1 |
| **Median [Q1, Q3]** | 4.0 [3.8, 4.3] |
| **Skewness / Kurtosis** | 0.42 / 1.49 |

**Clinical Note:** Normal 3.5-5.0 mmol/L. Critical for cardiac function.

---

## 7. Lipid Profile

### `cholesterol_mg_dl`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 28,855 (84.6%) |
| **Missing** | 5,242 (15.4%) |
| **Mean ± Std** | 188.00 ± 41.69 |
| **Range** | 62.0 - 813.0 |
| **Median [Q1, Q3]** | 185.0 [159.0, 213.0] |
| **Skewness / Kurtosis** | 0.79 / 3.56 |

**Clinical Note:** Desirable <200 mg/dL. Borderline 200-239 mg/dL. High ≥240 mg/dL.

---

## 8. Lifestyle Factors

### `alcohol_drinks_per_week`
| Attribute | Value |
|-----------|-------|
| **Type** | Continuous |
| **Non-Null** | 23,588 (69.2%) |
| **Missing** | 10,509 (30.8%) |
| **Mean ± Std** | 3.48 ± 8.07 |
| **Range** | 0.0 - 126.0 |
| **Median [Q1, Q3]** | 0.5 [0.0, 3.5] |
| **Skewness / Kurtosis** | 5.49 / 44.18 |

**Clinical Note:** Moderate drinking ≤14/week (men), ≤7/week (women). Heavy right skew.

---

### `smoking_status`
| Attribute | Value |
|-----------|-------|
| **Type** | Categorical |
| **Non-Null** | 14,336 (42.0%) |
| **Missing** | 19,761 (58.0%) ⚠️ |

| Category | Count | Percentage |
|----------|-------|------------|
| 3 (Never Smoked) | 8,227 | 57.4% |
| 1 (Current Smoker) | 4,814 | 33.6% |
| 2 (Former Smoker) | 1,295 | 9.0% |

**Clinical Note:** High missing rate due to questionnaire subset. Use masked loss.

---

# 🎯 TARGET VARIABLES

> **Important:** The following 8 columns are the prediction targets for the multi-task learning model. These columns should be treated specially during training (masked loss, class weights, etc.)

---

## 🎯 Task A: Cardiovascular Disease

### `has_cardiovascular_disease` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Binary Classification |
| **Task** | CVD Detection |
| **Non-Null** | 34,097 (100.0%) |
| **Missing** | 0 (0.0%) ✅ |

| Class | Count | Percentage |
|-------|-------|------------|
| **Healthy (0)** | 30,018 | **88.0%** |
| **Has CVD (1)** | 4,079 | **12.0%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **7.36:1** ⚠️ |
| **Recommended pos_weight** | 7.36 |

**Training Configuration:**
```python
# Weighted BCEWithLogitsLoss
pos_weight = torch.tensor([7.36])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Key Correlations:**
- `age`: 0.322 (strongest)
- `high_blood_pressure`: 0.303
- `body_mass_index`: 0.204

---

## 🎯 Task B: Metabolic Syndrome (5 Binary Components)

### `high_waist_circumference` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Binary Classification |
| **Task** | Metabolic Component 1 |
| **Non-Null** | 29,140 (85.5%) |
| **Missing** | 4,957 (14.5%) |

| Class | Count | Percentage |
|-------|-------|------------|
| **Normal (0)** | 11,810 | **40.5%** |
| **High Waist (1)** | 17,330 | **59.5%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **1.47:1** ✅ Balanced |
| **Recommended Strategy** | No weighting needed |

**Key Correlations:**
- `body_mass_index`: 0.640 (strongest)

---

### `high_triglycerides_mg_dl` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Binary Classification |
| **Task** | Metabolic Component 2 |
| **Non-Null** | 14,213 (41.7%) |
| **Missing** | 19,884 (58.3%) ⚠️ **HIGH MISSING** |

| Class | Count | Percentage |
|-------|-------|------------|
| **Normal (0)** | 11,228 | **79.0%** |
| **High Triglycerides (1)** | 2,985 | **21.0%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **3.76:1** |
| **Recommended pos_weight** | 3.76 |
| **Task Weight** | 0.5 (reduced due to missing) |

**Training Note:** Fasting-only subset. Use masked loss and reduced task weight.

---

### `low_hdl_mg_dl` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Binary Classification |
| **Task** | Metabolic Component 3 |
| **Non-Null** | 28,855 (84.6%) |
| **Missing** | 5,242 (15.4%) |

| Class | Count | Percentage |
|-------|-------|------------|
| **Normal HDL (0)** | 20,370 | **70.6%** |
| **Low HDL (1)** | 8,485 | **29.4%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **2.40:1** |
| **Recommended Strategy** | Mild weighting |

---

### `high_blood_pressure` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Binary Classification |
| **Task** | Metabolic Component 4 |
| **Non-Null** | 28,698 (84.2%) |
| **Missing** | 5,399 (15.8%) |

| Class | Count | Percentage |
|-------|-------|------------|
| **Normal BP (0)** | 18,370 | **64.0%** |
| **High BP (1)** | 10,328 | **36.0%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **1.78:1** ✅ Mild |
| **Recommended Strategy** | Mild weighting |

**Key Correlations:**
- `age`: 0.346 (strongest)

---

### `high_glucose_mg_dl` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Binary Classification |
| **Task** | Metabolic Component 5 |
| **Non-Null** | 13,816 (40.5%) |
| **Missing** | 20,281 (59.5%) ⚠️ **HIGH MISSING** |

| Class | Count | Percentage |
|-------|-------|------------|
| **Normal Glucose (0)** | 5,687 | **41.2%** |
| **High Glucose (1)** | 8,129 | **58.8%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **1.43:1** ✅ Balanced |
| **Task Weight** | 0.5 (reduced due to missing) |

**Training Note:** Fasting-only subset. Majority positive in available samples.

**Key Correlations:**
- `age`: 0.303

---

## 🎯 Task C: Kidney Function (Albuminuria Risk)

### `albuminuria_risk` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Multi-Class Classification (3 classes) |
| **Task** | Kidney Function Assessment |
| **Non-Null** | 30,171 (88.5%) |
| **Missing** | 3,926 (11.5%) |

| Class | Label | Count | Percentage |
|-------|-------|-------|------------|
| **0** | Normal (ACR <30) | 26,217 | **86.9%** |
| **1** | Microalbuminuria (30-300) | 3,265 | **10.8%** |
| **2** | Macroalbuminuria (>300) | 689 | **2.3%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **38:5:1** ⚠️ **EXTREME** |
| **Recommended Strategy** | Inverse frequency class weights |

**Training Configuration:**
```python
# Inverse frequency class weights
class_counts = torch.tensor([26217, 3265, 689], dtype=torch.float)
weights = 1.0 / class_counts
weights = weights / weights.sum() * 3  # Normalize

criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
# Use -1 for masked samples
```

> ⚠️ **Critical Warning:** Class 2 (Macroalbuminuria) has only 689 samples (2.3%). Without class weighting, the model will achieve 98% accuracy by always predicting Class 0.

**Key Correlations:**
- `creatinine_mg_dl`: 0.254
- `age`: 0.192

---

## 🎯 Task D: Liver Function

### `liver_dysfunction` 🎯 **[TARGET]**

| Attribute | Value |
|-----------|-------|
| **Type** | Binary Classification |
| **Task** | Liver Dysfunction Detection |
| **Non-Null** | 28,708 (84.2%) |
| **Missing** | 5,389 (15.8%) |

| Class | Count | Percentage |
|-------|-------|------------|
| **Normal (0)** | 24,373 | **84.9%** |
| **Dysfunction (1)** | 4,335 | **15.1%** |

| Metric | Value |
|--------|-------|
| **Imbalance Ratio** | **5.62:1** ⚠️ |
| **Recommended pos_weight** | 5.62 |

**Training Configuration:**
```python
pos_weight = torch.tensor([5.62])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Key Correlations:**
- `liver_ast_U_L`: 0.446 (strongest)
- `liver_ggt_U_L`: 0.305
- `body_mass_index`: 0.121

---

# High Correlations Summary

| Feature 1 | Feature 2 | Correlation |
|-----------|-----------|-------------|
| `gender` | `height_cm` | -0.677 |
| `body_mass_index` | `high_waist_circumference` 🎯 | 0.641 |
| `gender` | `hemoglobin_g_dl` | -0.512 |
| `liver_ast_U_L` | `liver_dysfunction` 🎯 | 0.446 |
| `liver_ast_U_L` | `liver_ggt_U_L` | 0.416 |
| `gender` | `uric_acid_mg_dl` | -0.403 |
| `height_cm` | `hemoglobin_g_dl` | 0.378 |
| `age` | `high_blood_pressure` 🎯 | 0.346 |
| `hemoglobin_g_dl` | `mean_corpuscular_volume_fL` | 0.329 |
| `age` | `has_cardiovascular_disease` 🎯 | 0.322 |

---

# Missing Values Analysis

## Columns with >50% Missing ⚠️

| Column | Missing | Percent | Impact | Action |
|--------|---------|---------|--------|--------|
| `high_glucose_mg_dl` 🎯 | 20,281 | **59.5%** | Fasting-only subset | Masked loss, task_weight=0.5 |
| `high_triglycerides_mg_dl` 🎯 | 19,884 | **58.3%** | Fasting-only subset | Masked loss, task_weight=0.5 |
| `smoking_status` | 19,761 | **58.0%** | Adult questionnaire gaps | Masked loss if used as input |

## Columns with 10-30% Missing

| Column | Missing | Percent |
|--------|---------|---------|
| `alcohol_drinks_per_week` | 10,509 | 30.8% |
| `heart_rate_bpm` | 6,622 | 19.4% |
| All liver markers | ~5,400 | ~16% |
| All kidney markers | ~5,400 | ~16% |
| Blood tests | ~4,600 | ~13.5% |

---

# Recommended Model Configuration

```python
config = {
    # Class weights for imbalanced targets
    'cvd_pos_weight': 7.36,
    'liver_pos_weight': 5.62,
    'triglycerides_pos_weight': 3.76,
    'albuminuria_class_weights': [0.073, 0.59, 2.8],  # Inverse frequency
    
    # Task weights (reduce for high-missing targets)
    'task_weights': {
        'cvd': 1.0,
        'metabolic_waist': 1.0,
        'metabolic_bp': 1.0,
        'metabolic_hdl': 1.0,
        'metabolic_trig': 0.5,      # 58% missing
        'metabolic_glucose': 0.5,   # 60% missing
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

# Critical Training Warnings

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

# Summary Table: All 8 Target Variables

| Target | Type | Classes | Imbalance | Missing % | Weight Strategy |
|--------|------|---------|-----------|-----------|-----------------|
| `has_cardiovascular_disease` 🎯 | Binary | 2 | 7.4:1 | 0% | `pos_weight=7.36` |
| `high_waist_circumference` 🎯 | Binary | 2 | ~1.5:1 | 14.5% | None needed |
| `high_blood_pressure` 🎯 | Binary | 2 | ~1.8:1 | 15.8% | Mild weighting |
| `low_hdl_mg_dl` 🎯 | Binary | 2 | ~2.4:1 | 15.4% | Mild weighting |
| `high_triglycerides_mg_dl` 🎯 | Binary | 2 | ~3.8:1 | **58.3%** | Weight + task weight |
| `high_glucose_mg_dl` 🎯 | Binary | 2 | ~1.4:1 | **59.5%** | Task weight only |
| `albuminuria_risk` 🎯 | 3-Class | 3 | 38:5:1 | 11.5% | Class weights |
| `liver_dysfunction` 🎯 | Binary | 2 | 5.6:1 | 15.8% | `pos_weight=5.62` |

---

**End of EDA Summary**

*Generated: December 26, 2025 from notebooks 01-07*
