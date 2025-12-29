---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Health Risks Prediction
### Deep Learning Model for: Heart | Diabetes | Kidney | Liver

---

# Our Neural Network

```
Patient Data                  🧠 Our Model                     Predictions
     ↓                              ↓                              ↓
Age, Weight,          →      [Learn Patterns]      →      ❤️ Heart Risk: 23%
Blood Pressure,                                           🍬 Diabetes Risk: 45%
Kidney, Liver Tests                                       🫘 Kidney Risk: 12%
                                                          🫀 Liver Risk: 8%
```

---

# The 3-Step Pipeline

```
            Step 1: ETL          Step 2: EDA          Step 3: MODEL

                📥                   🔍                   🤖
```

---

# Step 1: ETL 📥
- Source: **N**ational **H**ealth **a**nd **N**utrition **E**xamination **S**urvey (NHANES)
- **Folder:** `1. ETL/` - Collects and Cleans the data

---

# ETL Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  01_ingestion   │    │ 02_harmonize    │    │ 03_transform    │
│                 │    │                 │    │                 │
│  75 .xpt files  │───►│ Clinical rules  │───►│ MICE imputation │
│  15 categories  │    │ Gender-adjusted │    │ Train/Test split│
│  5 survey years │    │ thresholds      │    │ StandardScaler  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │
        ▼                      ▼                      ▼
   Raw NHANES            34,097 patients         ML-Ready Data
```

---

# ETL: Data Sources

| Category | Examples | Files |
|----------|----------|-------|
| Demographics | Age, Gender, Ethnicity | 5 |
| Body Measures | BMI, Height, Waist | 5 |
| Blood Tests | CBC, Biochemistry | 10 |
| Metabolic | Glucose, Cholesterol, HDL, Triglycerides | 20 |
| Organ Function | Kidney (ACR), Liver (ALT/AST) | 10 |
| Lifestyle | Smoking, Alcohol | 10 |

**Total: 75 files → 34,097 patients**

---

<!-- _class: lead -->

# Step 2: EDA 🔍
## Exploratory Data Analysis

---

# EDA: Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Patients** | 34,097 |
| **Features** | 21 inputs + 8 targets |
| **Age Range** | 20 - 80 years |
| **Mean BMI** | 29.7 (overweight) |

---

# EDA: Univariate Analysis

<!-- Shows distribution of each variable individually -->

| Finding | Impact |
|---------|--------|
| Age: uniform 20-80 | Good coverage |
| BMI: right-skewed | Some extreme obesity |
| Blood tests: ~15% missing | Need masked loss |

**Run:** `2. EDA/03_univariate_analysis.ipynb`
**Diagram:** Histograms of all 18 continuous features

---

# EDA: Class Imbalance Problem ⚠️

```
                    CVD Target Distribution

    ████████████████████████████████████████  88% Healthy
    █████  12% Has CVD

                    → Model will predict "healthy" for everyone!
                    → Solution: Weighted loss function
```

---

![bg contain](../assets/univariate-histograms.png)

---

# EDA: Bivariate Analysis

**Top correlations with Heart Disease:**

| Feature | Correlation | Meaning |
|---------|-------------|---------|
| Age | **0.32** | Older → more risk |
| Creatinine | 0.16 | Kidney link |
| Cholesterol | -0.14 | Inverse (treatment?) |

**Run:** `2. EDA/04_bivariate_analysis.ipynb`
**Diagram:** Boxplots of Age/BMI by CVD status

---

![bg contain](../assets/bivariate-heatmap.png)

---

# EDA: Outlier Analysis

```
            Z-Score vs IQR Outlier Detection

    alcohol_drinks_per_week    ████████████  11.9% outliers
    liver_ggt_U_L              █████████     8.8% outliers
    liver_ast_U_L              █████         5.2% outliers

    Decision: KEEP (medically valid extreme cases)
```

**Run:** `2. EDA/05_outlier_analysis.ipynb`

---

![bg contain](../assets/outlier-boxplots.png)

---

# EDA: Multivariate Analysis

<!-- Correlation heatmap showing feature relationships -->

**Key correlations found:**
- `liver_ast` ↔ `liver_ggt`: 0.42 (both liver markers)
- `height` ↔ `hemoglobin`: 0.38 (gender effect)
- `platelets` ↔ `WBC`: 0.32 (blood cell counts)

**Run:** `2. EDA/06_multivariate_analysis.ipynb`
**Diagram:** Correlation heatmap (15×15 matrix)

---

![bg contain](../assets/multivariate-correlation-heatmap.png)

---

# EDA: Key Findings Summary

| Challenge | Solution |
|-----------|----------|
| **58% missing** (glucose, triglycerides) | Masked loss |
| **7:1 class imbalance** (CVD) | Focal Loss + pos_weight |
| **38:1 rare class** (severe kidney) | Ordinal encoding |
| **High skewness** (liver markers) | Keep as valid medical cases |

---

<!-- _class: lead -->

# Step 3: Model 🤖
## Architecture & Training

---

# Model Architecture

![](../assets/simple-architecture.png)

---

<!-- _class: invert -->
![bg contain](../assets/neural-layer.png)

---

# Inside Each Neural Layer

- **Linear** - Weighted sum of inputs (learning)
- **BatchNorm** - Rescales numbers to average=0, spread=1 (stability)
- **LeakyReLU** - Keeps positives, scales negatives to 10% (non-linearity)
- **Dropout** - Randomly turns off 5% of neurons during training (prevent overfitting)

---

# Why This Design?

| Design Choice | Simple Explanation |
|---------------|-------------------|
| **Neural Network** | Can learn complex patterns |
| **Shared Backbone** | One brain learns features useful for all 4 predictions |
| **4 Separate Heads** | Each condition gets specialized output |
| **Multi-Task Learning** | Training on 4 tasks together makes each task better |

---

<!-- _class: lead -->

# Model Evaluation 📊

---

# Training Progress

<!-- Loss decreasing over epochs = model is learning -->

![](../training_loss_curve.png)

**Loss dropped from 1.9 → 0.77** (60% reduction)

---

# Results: Cardiovascular Disease ❤️

| Metric | Value | Meaning |
|--------|-------|---------|
| **ROC-AUC** | **0.83** | Good ranking ability |
| **Recall** | 55.8% | Catches 56% of sick patients |
| **Accuracy** | 83.5% | Overall correctness |

```
Confusion Matrix:     Predicted
                    Healthy   CVD
Actual  Healthy      5649     838
        CVD           375     473  ← Caught!
```

---

# Results: Metabolic Syndrome 🍬

| Component | Accuracy | ROC-AUC |
|-----------|----------|---------|
| **Waist** | **90.9%** | **0.97** |
| Triglycerides | 67.4% | 0.73 |
| HDL | 67.5% | 0.74 |
| Blood Pressure | 67.0% | 0.74 |
| Glucose | 65.4% | 0.69 |

**Overall Micro-F1: 0.72**

---

# Results: Kidney Function 🫘

| Class | Recall | Count |
|-------|--------|-------|
| **Normal** | **84.0%** | 5,681 |
| Microalbuminuria | 37.1% | 1,003 |
| Macroalbuminuria | 30.1% | 203 |

**Challenge:** Only 203 severe cases (2%) → hard to learn

---

# Results: Liver Dysfunction 🫀

| Metric | Value |
|--------|-------|
| **ROC-AUC** | **0.92** |
| **Accuracy** | **88.3%** |
| **Macro-F1** | **0.82** |

```
Confusion Matrix:     Predicted
                    Normal  Dysfunction
Actual  Normal       4884      279
        Dysfunction   501      977  ← Best performer!
```

---

# Overall Performance Summary

| Task | ROC-AUC | Macro-F1 | Verdict |
|------|---------|----------|---------|
| ❤️ Heart | 0.83 | 0.67 | ✅ Good |
| 🍬 Metabolic | 0.97* | 0.72 | ✅ Excellent (waist) |
| 🫘 Kidney | - | 0.50 | ⚠️ Needs work |
| 🫀 Liver | **0.92** | **0.82** | ✅ Best |

*Waist component only

---

# Key Metrics Explained

| Metric | What It Means | Our Goal |
|--------|---------------|----------|
| **ROC-AUC** | Can model rank sick vs healthy? | > 0.75 ✅ |
| **Recall** | Of all sick people, how many caught? | > 50% ✅ |
| **Macro-F1** | Balance across all classes | > 0.60 ✅ |

---

# Thank You!

**Files to explore:**
- `2. EDA/` - All 7 analysis notebooks
- `3. Model/05_evaluate.py` - Evaluation code
- `3. Model/MODEL_EVALUATION_REPORT.md` - Full results

