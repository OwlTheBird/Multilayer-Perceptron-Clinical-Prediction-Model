# ETL Pipeline

> ⚠️ **Known Issue:** The current implementation has a **flow order problem**. Stage 3 (Transformation.py) splits data into train/test **before** any EDA is performed. This is incorrect - EDA should happen first to see the full dataset and make informed cleaning decisions.
>
> **Workaround:** We're performing EDA separately on the full dataset from `nhanes_1st.db` (see [eda-analysis.md](eda-analysis.md)). The pipeline will be refactored after team coordination.

## What is an ETL Pipeline?

**ETL = Extract, Transform, Load**

An **ETL pipeline** is a process that moves data from source to destination through three key stages:
- **Extract:** Collect raw data from various sources
- **Transform:** Clean, standardize, and prepare the data
- **Load:** Save the processed data to a database or storage

Think of it like an assembly line in a factory - raw materials (data) go through different stations (ETL stages) to become a finished product (ML-ready dataset).

## The Pipeline Stages

### Stage 1: Extract + Load
**File:** `Ingestion.py`

**Input:** `.xpt` files (NHANES Survey Data 2013-2023)
- Demographics (age, gender, race)
- Body measurements (height, weight, BMI)
- Blood tests (cholesterol, glucose)
- Health questionnaires (smoking, drinking, heart problems)

**What happens:** Extract raw survey files from multiple sources and Load them into the database.

**Output:** `nhanes_1st.db` (raw tables organized by topic)

---

### Stage 2: Transform (Part 1 - Harmonization)
**File:** `Harmonization.py`

**Input:** Raw tables in `nhanes_1st.db`

**What happens:** Standardize inconsistent schemas and formats across different survey years.

**Problems we solve:**

| Problem | Solution |
|---------|----------|
| 2013-2016 measured blood pressure manually, 2017+ used automatic machines | Combine both methods into one standard measurement |
| Triglycerides column changed names between years | Rename all to the same column name |
| Alcohol questions changed between survey cycles | Convert all answers to "drinks per week" |

**Example:**
```
Before: BPXSY1 (manual BP) and BPXOSY1 (automatic BP) are separate
After:  BPXSY1 (unified BP reading)
```

**Output:** `nhanes_1st.db` (updated with cleaned, unified tables)

---

### Stage 3: Transform (Part 2 - Feature Engineering) + Load
**File:** `Transformation.py`

**Input:** Cleaned tables in `nhanes_1st.db`

**What happens:** Create target variables, engineer features, preprocess data, and Load into final ML database.

**3.1 Create Target Variables:**
- **Cardiovascular:** Did patient have heart disease? (Yes/No)
- **Metabolic:** Which metabolic syndrome criteria are met? (5 Yes/No labels)
- **Kidney:** Albumin-to-Creatinine Ratio (log-transformed)
- **Liver:** ALT enzyme level (log-transformed)

**3.2 Feature Preprocessing:**
- Remove rows where ALL targets are missing
- Split into training (80%) and test (20%) sets
- Scale continuous features (age, BMI, etc.)
- Encode categorical features (gender, race)
- Impute missing values (KNN)

**3.3 Load to Final Database:**
Save processed data ready for machine learning.

**Output:** `ML_data.db` with two tables:
- `training_set` (80% of data)
- `testing_set` (20% of data)

---

## Summary

```
┌─────────────────┐
│  Raw .xpt Files │ (NHANES Survey Data 2013-2023)
└────────┬────────┘
         │
         ↓ [Stage 1: Extract + Load]
         ↓ Ingestion.py
┌─────────────────┐
│ nhanes_1st.db   │ (Raw tables)
└────────┬────────┘
         │
         ↓ [Stage 2: Transform Part 1 - Harmonization]
         ↓ Harmonization.py
┌─────────────────┐
│ nhanes_1st.db   │ (Cleaned, unified tables)
└────────┬────────┘
         │
         ↓ [Stage 3: Transform Part 2 + Load]
         ↓ Transformation.py
┌─────────────────┐
│  ML_data.db     │ (training_set + testing_set)
└─────────────────┘
```

The pipeline ensures that messy, inconsistent real-world data becomes clean, reliable data that a machine learning model can actually learn from.

---

**Next:** Read [eda-analysis.md](eda-analysis.md) to see what we discovered during data analysis.
