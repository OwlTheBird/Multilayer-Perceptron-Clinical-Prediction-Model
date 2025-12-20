---
marp: true
theme: default
paginate: true
backgroundColor: #fff
---

# Multilayer Perceptron Clinical Prediction Model
### Using Machine Learning to Predict Health Risks

---

# What Does Our Project Do?

Imagine you go to the doctor and they take some measurements:
- Your **height** and **weight**
- Your **blood pressure**
- Some **blood tests**

Our model predicts:
**"Is this person at risk for heart disease, diabetes, kidney problems, or liver issues?"**

---

# Think of it Like a Smart Doctor Assistant

```
Patient Data (Input)          →    Our Model    →    Health Predictions (Output)
                                   (The Brain)

Age, Weight, Blood Pressure        🧠              ❤️ Heart Risk: 23%
Blood Sugar, Cholesterol                           🍬 Diabetes Risk: 45%
Kidney Tests, Liver Tests                          🫘 Kidney Risk: 12%
                                                   🫀 Liver Risk: 8%
```

---

# Where Does the Data Come From?

## NHANES Dataset
- **N**ational **H**ealth **a**nd **N**utrition **E**xamination **S**urvey
- Real health data from **thousands of Americans**
- Collected by the US government (2013-2023)
- **70 data files** with different health measurements

Think of it as a giant spreadsheet with health info from real people!

---

# The 3-Step Pipeline

Our project works in **3 main steps**:

```
Step 1: COLLECT        Step 2: CLEAN         Step 3: PREDICT
   📥                     🧹                     🤖

Get raw data         Fix messy data        Train the model
from 70 files        & organize it         to make predictions
```

---

# Step 1: Collect the Data (Ingestion)

**File:** `Ingestion.py`

What it does:
1. Opens each of the 70 health data files
2. Picks out the important columns we need
3. Saves everything into one database

**Analogy:** Like collecting puzzle pieces from 70 different boxes and putting them in one big box

---

# What Data Do We Collect?

| Category | Examples |
|----------|----------|
| **Demographics** | Age, Gender, Race |
| **Body Measurements** | Height, Weight, Waist size |
| **Blood Pressure** | Systolic, Diastolic (multiple readings) |
| **Blood Tests** | Cholesterol, Blood Sugar, Kidney markers |
| **Lifestyle** | Smoking, Alcohol, Physical activity |

---

# Step 2: Clean & Transform

**Files:** `Harmonization.py` and `Transformation.py`

### Problem: The data is messy!
- Some people have missing values
- Different years use different column names
- Numbers need to be on the same scale

### Solution: Clean it up!
- Fill in missing values smartly
- Standardize all the names
- Scale numbers so they're comparable

---

# Creating the Target Labels

We need to tell the model: "This person IS at risk" or "This person is NOT at risk"

**Example - Diabetes Risk:**
```
IF blood sugar > 126 mg/dL  →  Label = 1 (At Risk)
IF blood sugar ≤ 126 mg/dL  →  Label = 0 (Not at Risk)
```

These rules come from **real medical guidelines**!

---

# Step 3: The Neural Network (Our Brain)

**File:** `model.py`

A neural network is like a **chain of decisions**:

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
        (256 nodes)  (192 nodes)  (128 nodes)

Each layer asks: "What patterns do I see?"
```

---

# Multi-Task Learning: One Brain, Four Jobs

Our model is special - it predicts **4 things at once**:

```
                    Shared Brain
                        ↓
            ┌─────┬─────┬─────┬─────┐
            ↓     ↓     ↓     ↓     ↓
          Heart  Diabetes Kidney Liver
          Head    Head    Head   Head
```

**Why?** Learning multiple things together makes the model smarter!
(Like how learning math helps you with physics)

---

# The Magic of Masks

**Problem:** Not every patient has all their test results

**Solution:** Use masks!

```
Patient A: Heart ✓  Diabetes ✓  Kidney ✗  Liver ✓
Mask:      [1]      [1]         [0]       [1]

When training:
- Use Heart, Diabetes, Liver data
- Ignore (mask out) the missing Kidney data
```

The model learns from what it CAN see!

---

# Training the Model

```python
for each round (epoch):
    for each batch of patients:
        1. Make predictions
        2. Check how wrong we were (loss)
        3. Adjust the brain to be less wrong

# After 20 rounds, save the trained model
```

**Loss** = How wrong the model is (lower is better!)

---

# The Web App (Streamlit)

**File:** `streamlit_app.py`

A user-friendly interface where you can:
1. Enter a patient's health data
2. Click a button
3. See the predicted health risks!

```
[Age: 45]  [Weight: 180]  [Blood Pressure: 130/85]

                [PREDICT]

"Heart Disease Risk: 34%"
```

---

# Project File Structure

```
📁 Project
├── 📁 1. Extraction and Loading/    ← Step 1: Collect
│   ├── Ingestion.py
│   └── Raw Data/ (70 files)
│
├── 📁 2. Transformation/            ← Step 2: Clean
│   ├── Harmonization.py
│   └── Transformation.py
│
├── 📁 model/                        ← Step 3: Predict
│   ├── model.py (the neural network)
│   ├── train.py (training logic)
│   └── streamlit_app.py (web interface)
│
└── 📁 databases/                    ← Our data storage
    ├── nhanes_1st.db (raw data)
    └── ML_data.db (clean data)
```

---

# Key Technologies Used

| Tool | What It Does |
|------|--------------|
| **Python** | The programming language |
| **Pandas** | Reading and manipulating data |
| **PyTorch** | Building the neural network |
| **SQLite** | Storing data in databases |
| **DuckDB** | Fast data transformations |
| **Scikit-learn** | Data preprocessing |
| **Streamlit** | Building the web app |

---

# Thank You!
