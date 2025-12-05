# Model Card

## What is This Model?

This is a **multi-task neural network** that acts like a medical screening assistant. You give it patient health information, and it predicts risks in 4 different health areas at once.

Think of it as one brain with 4 specialized thinking areas - each focused on a different type of health problem.

## Model Details

**Model Type:** Multilayer Perceptron (MLP) - a type of neural network

**Architecture:**
- 1 Shared Body (learns general health patterns)
- 4 Task-Specific Heads (each specializes in one prediction)

**Training Data:** NHANES (National Health and Nutrition Examination Survey) 2013-2023

**Input Features (21 total):**
- Demographics: age, gender, race, income
- Body measurements: BMI, height, waist circumference, pulse
- Blood tests: cholesterol, glucose, kidney/liver enzymes, blood cell counts
- Lifestyle: alcohol consumption, smoking status

## What Does It Predict?

### Head 1: Cardiovascular Risk
**Predicts:** Has the patient ever had major heart disease?

**Output:** Probability (0-100%)

**Includes:** Heart attack, stroke, heart failure, coronary heart disease, angina

**Task Type:** Binary Classification (Yes/No)

---

### Head 2: Metabolic Syndrome
**Predicts:** Which metabolic syndrome criteria does the patient meet?

**Outputs:** 5 separate probabilities (0-100%) for:
1. Abdominal obesity (large waist)
2. High triglycerides
3. Low HDL cholesterol (good cholesterol)
4. High blood pressure
5. High fasting glucose

**Task Type:** Multi-Label Classification (can have multiple "Yes" answers)

---

### Head 3: Kidney Health
**Predicts:** Urine Albumin-to-Creatinine Ratio (UACR)

**Output:** A number indicating kidney damage level

**What it means:**
- Low values = healthy kidneys
- High values = kidney damage

**Task Type:** Regression (predicts a continuous number)

---

### Head 4: Liver Health
**Predicts:** ALT enzyme level

**Output:** A number indicating liver inflammation

**What it means:**
- Low ALT = healthy liver
- High ALT = liver damage/inflammation

**Task Type:** Regression (predicts a continuous number)

## How Good Is It?

**Evaluation Metrics** (how we measure performance):

| Head | Metric | What It Means |
|------|--------|---------------|
| Cardiovascular | AUROC | How well it separates healthy vs. at-risk patients |
| Metabolic | LRAP / F1-Score | How accurately it identifies each syndrome component |
| Kidney | R² and RMSE | How close predictions are to actual kidney damage |
| Liver | R² and RMSE | How close predictions are to actual liver enzyme levels |

**Note:** Actual performance numbers will be added after model training is complete.

## How Does It Learn?

**Loss Function:** A weighted combination of errors from all 4 heads

```
Total Loss = λ₁ × Cardio Loss + λ₂ × Metabolic Loss + λ₃ × Kidney Loss + λ₄ × Liver Loss
```

- Cardio & Metabolic use Binary Cross-Entropy (good for Yes/No predictions)
- Kidney & Liver use Mean Squared Error (good for number predictions)
- The λ weights are adjusted to balance the 4 tasks

## When to Use This Model

### ✅ Good Use Cases:
- Screening large populations for health risks
- Identifying patients who need follow-up testing
- Research on multi-system health interactions
- Educational demonstrations of multi-task learning

### ❌ Do NOT Use For:
- Final medical diagnosis (this is a screening tool only!)
- Treatment decisions without doctor consultation
- Populations very different from NHANES (e.g., non-US populations)
- Individual medical advice

## Limitations

1. **Not a Doctor:** This model screens for risks but cannot diagnose. Always consult medical professionals.

2. **Training Data Bias:** Trained only on US survey data from 2013-2023. May not generalize well to:
   - Other countries
   - Rare diseases
   - Age groups or demographics underrepresented in NHANES

3. **Missing Data:** Model performance depends on having quality input features. Garbage in = garbage out.

4. **Correlation ≠ Causation:** The model finds patterns, but doesn't understand WHY those patterns exist.

5. **No Longitudinal Data:** NHANES is cross-sectional (one snapshot in time), so the model can't track how health changes over time.

## Ethical Considerations

- **Privacy:** Model should only be used on de-identified data
- **Fairness:** Performance should be evaluated across demographic groups to ensure no bias
- **Transparency:** Users must know this is an AI prediction, not a medical diagnosis
- **Accountability:** Decisions made using this model should be overseen by qualified healthcare professionals

## Technical Details

**Framework:** PyTorch (planned)

**Preprocessing:**
- StandardScaler for continuous features
- One-Hot Encoding for categorical features (gender, race)
- KNN Imputation for missing values

**Loss Functions:**
- Binary Cross-Entropy (BCE) for classification heads
- Mean Squared Error (MSE) for regression heads

**Hyperparameters:** (To be determined during training)
- Learning rate
- Number of layers
- Hidden layer sizes
- Task loss weights (λ₁, λ₂, λ₃, λ₄)
- Batch size
- Number of epochs

## Version History

**Version 0.1 (Current):** Data pipeline complete, model architecture designed, training not yet started

## Contact & Citation

This is an academic project. For questions about the model or to report issues, please refer to the project repository.

---

**Done!** You've completed all the documentation. Return to [START-HERE.md](START-HERE.md) or check the main [README.md](../README.md) for project details.
