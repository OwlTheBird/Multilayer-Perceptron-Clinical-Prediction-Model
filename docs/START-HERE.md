# Welcome to the Documentation

This folder contains documentation to help you understand the Multi-Task Clinical Prediction Model project.

## What is this project?

This project builds a machine learning model that can predict health problems by looking at patient data. It's like having a doctor's assistant that can spot potential issues in 4 different areas of health at once.

## How to read these docs

Please read the documentation files in this order:

### 1. [architecture.md](architecture.md) - **Start Here First**
See the big picture of how all the pieces fit together - the databases, the code, and the model.

**Why read this first?** Get the 30,000-foot view of the entire system before diving into details.

### 2. [etl-pipeline.md](etl-pipeline.md) - **Read Second**
Learn how raw health survey data is extracted, cleaned, and transformed into a machine learning-ready format.

**Why read this second?** Now that you understand the big picture, dive into the details of the ETL (Extract, Transform, Load) process.

### 3. [eda-analysis.md](eda-analysis.md) - **Read Third**
Explore the data analysis findings - what we discovered about data quality, outliers, and feature relationships.

**Why read this third?** Understanding the data characteristics helps you see why certain modeling decisions were made.

### 4. [model-card.md](model-card.md) - **Read Last**
Understand what the machine learning model actually predicts and how well it works.

**Why read this last?** With knowledge of the system, data pipeline, and data insights, you can now understand what the model does and its limitations.

## Quick Questions

**What does this model predict?**
- Heart disease risk
- Metabolic syndrome components
- Kidney health (albumin-creatinine ratio)
- Liver health (ALT enzyme levels)

**What data does it use?**
Real health survey data from NHANES (National Health and Nutrition Examination Survey) from 2013-2023.

**How does it work?**
It uses a neural network (multilayer perceptron) with one shared "body" that learns general health patterns, and 4 specialized "heads" that each focus on one type of prediction.
