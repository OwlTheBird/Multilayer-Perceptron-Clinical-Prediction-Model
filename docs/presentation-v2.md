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

<!-- _class: lead -->

# Step 3: The Neural Network
## A Deep Dive into Our Model Architecture

---

# Why a Neural Network? The Alternatives

| Model | Pros | Cons | Why Not for Us |
|-------|------|------|----------------|
| **Logistic Regression** | Simple, interpretable | Only linear relationships | Health data has complex non-linear patterns |
| **Random Forest** | Handles non-linearity | Separate model per task | Can't share learning across tasks |
| **XGBoost** | High accuracy | Same as Random Forest | No natural multi-task support |
| **Neural Network** | Non-linear + multi-task capable | Needs more data, less interpretable | **Best fit for our problem** |

---

# Why Neural Networks Win Here

**The key reason:** We're predicting 4 related health outcomes simultaneously.

- Heart disease, diabetes, kidney issues, and liver problems **share underlying causes**
- High blood pressure affects the heart AND kidneys
- Obesity affects metabolic syndrome AND liver function
- A neural network can **learn these shared patterns once** and use them for all 4 predictions

Traditional models would need 4 separate models, each learning from scratch.

---

# What is a Multilayer Perceptron (MLP)?

An MLP is the simplest type of neural network:

```
Input Features → Hidden Layer 1 → Hidden Layer 2 → Hidden Layer 3 → Output
    (27)            (256)            (192)            (128)          (8)
```

**Each layer:**
1. Takes numbers in
2. Multiplies them by learned weights
3. Adds them up
4. Applies an activation function (introduces non-linearity)
5. Passes result to next layer

---

# Why MLP Over Other Neural Network Types?

| Type | Best For | Our Data |
|------|----------|----------|
| **CNN** | Images, spatial patterns | ❌ We have tabular data, no spatial structure |
| **RNN** | Sequences, time series | ❌ Each patient is independent, no time dependency |
| **Transformer** | Text, very long sequences | ❌ Overkill for 27 features |
| **MLP** | Tabular data | ✅ Perfect fit |

**Bottom line:** MLPs are designed for structured tabular data. Using a CNN here would be like using a hammer to turn a screw.

---

# Our Architecture: Hard Parameter Sharing

```
        27 Input Features
              ↓
        [Shared Backbone]
         256 → 192 → 128     ← All 4 tasks share these layers
              ↓
    ┌────┬────┬────┬────┐
    ↓    ↓    ↓    ↓    ↓
  Heart Meta Kidney Liver   ← Each task has its own "head"
   (1)  (5)   (1)   (1)
```

**"Hard" parameter sharing** = The shared layers have the **exact same weights** for all tasks.

---

# Why Share Parameters?

## Benefit 1: Regularization Effect
- Each task acts as a **constraint** on the others
- Prevents overfitting because the model can't memorize task-specific noise
- Forces the model to learn **general health patterns**, not quirks

## Benefit 2: Data Efficiency
- If Task A has 10,000 samples and Task B has only 2,000 samples
- Task B benefits from the patterns learned from Task A's larger dataset
- The shared layers get trained on ALL available data

---

# Why Share Parameters? (continued)

## Benefit 3: Faster Training
- Train one shared backbone instead of 4 separate networks
- ~83,000 parameters total vs ~200,000+ if fully separate

## The Trade-off (Honesty Check)
- If tasks are **unrelated**, sharing can hurt performance (negative transfer)
- Our tasks ARE related (all cardiovascular/metabolic health), so sharing helps
- If tasks were "predict health" + "predict favorite color" → sharing would fail

---

# Layer Sizes: 256 → 192 → 128

**Why these specific numbers?**

```
Input (27) → 256 → 192 → 128 → Heads
             ↑      ↑      ↑
          Expand  Gradual  Compress
                 narrowing
```

1. **256 first**: Expand to capture complex feature interactions
2. **Gradual decrease**: Forces the network to compress information, keeping only what matters
3. **128 final**: A compact representation before task-specific heads

**Why not bigger?** More parameters = more risk of overfitting with ~50K samples.

---

# Activation Function: LeakyReLU(0.1)

**What it does:** Introduces non-linearity (allows learning curved patterns)

```
Standard ReLU:     LeakyReLU(0.1):
     /                  /
    /                  /
---+              ----+
   0               slight slope (0.1x)
```

**Why LeakyReLU over ReLU?**
- ReLU outputs 0 for all negative inputs → neurons can "die" (stop learning)
- LeakyReLU keeps a small gradient (0.1) for negative values
- Clinical data often has near-zero values after standardization → prevents dead neurons

---

# Dropout: 0.2 (20%)

**What it does:** Randomly "turns off" 20% of neurons during training

```
Training:          Without Dropout:
[●][○][●][●][○]    [●][●][●][●][●]
  ↓  ↓  ↓           all neurons active
Only 80% active     → may memorize training data
```

**Why 20%?**
- Too low (5%): Not enough regularization, overfitting risk
- Too high (50%): Too much information lost, underfitting
- 20%: Sweet spot for medium-sized networks with tabular data

---

# BatchNorm: Why Normalize?

**Problem:** During training, the distribution of inputs to each layer keeps changing (internal covariate shift)

**BatchNorm solution:** Normalize each layer's inputs to mean=0, std=1

```python
self.input_bn = nn.BatchNorm1d(num_continuous)  # On input
nn.BatchNorm1d(256)  # After each layer
```

**Benefits:**
- Faster training (can use higher learning rates)
- More stable gradients
- Slight regularization effect

---

# The Four Task Heads

| Head | Task Type | Output | Loss Function |
|------|-----------|--------|---------------|
| **Cardio** | Binary Classification | 1 probability | BCE (Binary Cross-Entropy) |
| **Metabolic** | Multi-Label Classification | 5 probabilities | BCE per label |
| **Kidney** | Regression | 1 continuous value | MSE (Mean Squared Error) |
| **Liver** | Regression | 1 continuous value | MSE |

Each head is just a single linear layer: `nn.Linear(128, output_size)`

---

# Why Raw Logits? (No Sigmoid in Forward Pass)

```python
out_cardio = self.head_cardio(z)  # Raw numbers, not probabilities
```

**Seems wrong?** Classification should output probabilities (0-1)...

**Actually correct!** We use `BCEWithLogitsLoss` which:
1. Applies sigmoid internally
2. Computes loss in log-space
3. **More numerically stable** than `sigmoid → BCE` separately

This avoids floating-point precision issues with very small probabilities.

---

# The Masking System: Handling Missing Labels

**Real-world problem:** Not every patient has all 4 diagnoses available

**Naive solution:** Only use patients with ALL labels → Lose 60% of data!

**Our solution:** Use masks to ignore missing labels in loss calculation

```python
mask = [1, 1, 0, 1]  # Patient has Heart, Metabolic, Liver but NOT Kidney
loss = (loss_heart × 1) + (loss_meta × 1) + (loss_kidney × 0) + (loss_liver × 1)
```

The gradient from Kidney is zeroed out → model still learns from other tasks.

---

# Masking: The Math

```python
def masked_bce_loss(pred, target, mask):
    loss = BCE(pred, target)      # Calculate loss for all
    masked_loss = loss * mask      # Zero out invalid ones
    return masked_loss.sum() / mask.sum()  # Average over valid only
```

**Why divide by `mask.sum()`?**
- If we divided by total samples, batches with more missing data would have smaller gradients
- Dividing by valid count keeps gradient magnitude consistent

---

# Multi-Task vs Single-Task: Real Comparison

| Approach | Parameters | Data Used | Performance |
|----------|------------|-----------|-------------|
| **4 Separate MLPs** | ~200K | Each task uses only its own labels | Baseline |
| **Multi-Task (Ours)** | ~83K | All tasks share backbone training | Usually 5-15% better |

**When does Multi-Task fail?**
- Tasks are completely unrelated
- One task dominates and hurts others (task imbalance)
- Not enough shared signal between tasks

**Our case:** All 4 tasks relate to cardiovascular/metabolic health → sharing helps.

---

# Potential Improvement: Task Weighting

**Current approach:** All tasks contribute equally to loss
```python
total_loss = loss_cardio + loss_metabolic + loss_kidney + loss_liver
```

**Problem:** MSE (regression) and BCE (classification) have different scales
- MSE might be 0.5-2.0
- BCE might be 0.3-0.7
- Regression tasks could dominate!

**Better approach (not implemented):** Learned task weights or uncertainty weighting

---

# What We Don't Have (Areas for Improvement)

| Missing | Impact | Why It Matters |
|---------|--------|----------------|
| **Validation Set** | Can't detect overfitting during training | Model might memorize training data |
| **Early Stopping** | Train for fixed 20 epochs | Might train too long or not enough |
| **Learning Rate Scheduling** | Fixed LR throughout | Could converge faster with decay |

These are documented limitations, not fundamental flaws.

---

# Summary: Why This Architecture?

1. **MLP** → Best for tabular data (not images/text)
2. **Multi-Task with Hard Sharing** → Related tasks benefit from shared learning
3. **256→192→128** → Balance between capacity and overfitting risk
4. **LeakyReLU** → Prevents dead neurons in clinical data
5. **Dropout 20%** → Standard regularization for this size
6. **Masking** → Use ALL available data, even with missing labels
7. **Raw Logits** → Numerically stable loss computation

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
