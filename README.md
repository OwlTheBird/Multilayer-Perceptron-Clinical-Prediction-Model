# Multi-Task Clinical Prediction Model

[cite_start]This project implements a multi-task deep learning model to simultaneously predict clinical outcomes across four distinct health domains: cardiovascular, metabolic, kidney, and liver health[cite: 3]. [cite_start]It uses a unified Multilayer Perceptron (MLP) architecture trained on the National Health and Nutrition Examination Survey (NHANES) dataset[cite: 6, 11]. [cite_start]The goal is to create a single, efficient model that learns a holistic representation of a patient's health to predict multiple outcomes at once[cite: 3, 4].

---

## 🏗️ Model Architecture

[cite_start]The model uses a multi-task, multi-output MLP architecture designed to learn generalized health patterns from a complete patient feature vector[cite: 6].

* **Shared Body**: A series of dense layers that processes the input features to learn a compressed, holistic representation of a patient's overall health status.
* **Task-Specific Heads**: Four independent sets of dense layers that branch off from the shared body. Each head is a specialist, mapping the shared health representation to a specific clinical target.



---

## 🎯 Prediction Tasks & Heads

The model is trained on four specific clinical tasks, each with its own dedicated head.

### 1. Cardiovascular Head ❤️
* **Goal**: Classify patients based on a prior diagnosis of major cardiovascular disease (CVD).
* **Task Type**: Binary Classification.
* **Target Variable**: A binary label derived from NHANES questionnaire data indicating a history of congestive heart failure, coronary heart disease, angina, a heart attack, or a stroke.
* **Output**: A single logit representing the probability of a positive CVD history.

### 2. Metabolic Head 🧬
* **Goal**: Identify which specific components of metabolic syndrome are present in a patient.
* **Task Type**: Multi-Label Classification.
* **Target Variable**: Five independent binary labels for each of the NCEP-ATP III criteria (abdominal obesity, high triglycerides, low HDL cholesterol, high blood pressure, high fasting glucose).
* **Output**: A vector of 5 logits, where each logit represents the probability that the patient meets a specific criterion.

### 3. Kidney Head երի
* **Goal**: Estimate the degree of kidney damage by predicting a key biomarker.
* **Task Type**: Regression.
* **Target Variable**: The Urine Albumin-to-Creatinine Ratio (UACR), a primary indicator of kidney damage, which is log-transformed to stabilize variance.
* **Output**: A single continuous value predicting log(UACR).

### 4. Liver Head 🧪
* **Goal**: Estimate the level of liver inflammation by predicting a key liver enzyme.
* **Task Type**: Regression.
* **Target Variable**: The blood level of Alanine Aminotransferase (ALT), a standard biomarker for liver cell injury, which is log-transformed.
* **Output**: A single continuous value predicting log(ALT).

---

## 📊 Dataset

The model is trained on data from the **National Health and Nutrition Examination Survey (NHANES)**, specifically using the 2013-2023 cycles.

---

## ⚙️ Technical Details

### Loss Function
[cite_start]The total loss is a weighted sum of the individual losses from each task head[cite: 45]. [cite_start]It combines Binary Cross-Entropy (BCE) for the classification tasks and Mean Squared Error (MSE) for the regression tasks[cite: 46, 47]. [cite_start]The weights (${\lambda_1, \lambda_2, \lambda_3, \lambda_4}$) are critical hyperparameters to be tuned[cite: 48].

$$L_{total} = \lambda_1 L_{BCE, cardio} + \lambda_2 L_{BCE, metabolic} + \lambda_3 L_{MSE, kidney} + \lambda_4 L_{MSE, liver}$$

### Evaluation Metrics
[cite_start]Each head is evaluated with a specific, appropriate metric[cite: 49].
* [cite_start]**Cardio**: Area Under the Receiver Operating Characteristic Curve (AUROC)[cite: 50].
* [cite_start]**Metabolic**: Label-Ranking Average Precision (LRAP) or Macro-Averaged F1-Score[cite: 51].
* [cite_start]**Kidney & Liver**: R-squared ($R^2$) and Root Mean Squared Error (RMSE)[cite: 52].
