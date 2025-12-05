# System Architecture

## What is Architecture?

Architecture is the **blueprint** of how our system is built. Just like a house has a blueprint showing where the kitchen, bedrooms, and bathrooms go, our project has a structure showing how all the parts work together.

## The Big Picture

```
┌──────────────┐
│  Raw Data    │  NHANES Survey Files (.xpt)
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  ETL Pipeline│  Clean & Prepare Data
│  (3 Stages)  │
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  Databases   │  Store Data
└──────┬───────┘
       │
       ↓
┌──────────────┐
│  ML Model    │  Make Predictions
│  (4 Heads)   │
└──────────────┘
```

## Project Structure

```
Multilayer-Perceptron-Clinical-Prediction-Model/
│
├── 1. Extraction and Loading Process/
│   └── Ingestion.py              ← Loads raw survey files into database
│
├── 2. Transformation/
│   ├── Harmonization.py          ← Cleans and standardizes data
│   └── Transformation.py         ← Prepares data for ML model
│
├── databases/
│   ├── nhanes_1st.db            ← Raw + cleaned data
│   └── ML_data.db               ← Ready-to-train data
│
└── docs/                         ← You are here!
    ├── START-HERE.md
    ├── architecture.md
    ├── etl-pipeline.md
    ├── eda-analysis.md
    └── model-card.md
```

## Components Explained

### 1. Data Storage (Databases)

**nhanes_1st.db** - Central data warehouse with raw and cleaned survey data organized by topic (Demographics, Blood Tests, etc.)

**ML_data.db** - Final processed data split into training and testing sets, ready for the model

### 2. Data Processing Scripts

Three Python scripts handle the data pipeline:
- **Ingestion.py** - Loads raw survey files into the database
- **Harmonization.py** - Standardizes inconsistent formats across survey years
- **Transformation.py** - Prepares final features and creates prediction targets

### 3. Machine Learning Model

A **Multi-Task Multilayer Perceptron** (neural network) with one shared body and 4 specialized heads:

```
Input Layer (Patient Features)
       ↓
Shared Body (learns general health patterns)
   ╱   ╲   ╲   ╲
  ↓    ↓    ↓    ↓
Head1 Head2 Head3 Head4
  │    │     │     │
  ↓    ↓     ↓     ↓
Cardio Metabolic Kidney Liver
```

Each head specializes in predicting one health outcome.

## Key Technologies

**Python** • **SQLite** • **Pandas** • **scikit-learn** • **PyTorch** (planned)

## Design Principles

**Modular:** Each script does one job well - easier to debug and update

**Database-Centric:** Data is saved at each stage - can inspect quality and restart from any point

**Multi-Task Learning:** One model predicts 4 outcomes - more efficient than separate models

---

**Next:** Read [etl-pipeline.md](etl-pipeline.md) to learn how data flows through the ETL pipeline in detail.
