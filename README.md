# 🧳 Travel Insurance Prediction & Analytics System

An end-to-end Data Science and Machine Learning platform analyzing customer demographic profiles, household financial metrics, travel history, and health risk factors to forecast travel insurance purchase intent using Python.

---

## 📌 Project Overview

This repository contains data preprocessing pipelines, feature engineering, exploratory data analysis (EDA), and machine learning models designed to extract actionable risk analytics and classify potential travel insurance buyers from customer dataset records.

### 🎯 Key Objectives

* **Data Cleansing & Transformation:** Load raw customer data, drop redundant index columns, standardize target labels to `target`, and eliminate duplicate records to ensure model integrity (reducing initial 1,987 rows to 1,249 unique records).
* **Domain Feature Engineering:** Derive interaction and financial features, including household per-capita income (`income_per_family_member`), risk profile indicators (`high_risk_traveler`), and income threshold brackets (`is_high_income`).
* **Exploratory Data Analysis (EDA):** Generate comprehensive visual distributions across all 12 feature columns to identify key drivers of insurance adoption.
* **Predictive Outcome Modeling:** Train, tune, and evaluate classification algorithms (Logistic Regression, Random Forest, and XGBoost) to forecast customer purchasing decisions.

---

## 🛠️ Dataset Architecture & Features

The dataset incorporates customer demographics, employment characteristics, health conditions, and travel background:

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `age` | Numerical | Age of the customer. |
| `employment_type` | Categorical | Employment sector (`Government Sector` or `Private Sector/Self Employed`). |
| `graduateornot` | Categorical | Educational qualification status (`Yes` / `No`). |
| `annualincome` | Numerical | Customer's annual income in local currency. |
| `familymembers` | Numerical | Total number of family members in the household. |
| `chronicdiseases` | Binary | Presence of chronic illness (`1` = Yes, `0` = No). |
| `frequentflyer` | Categorical | Frequent flyer membership status (`Yes` / `No`). |
| `evertravelledabroad` | Categorical | History of international travel (`Yes` / `No`). |
| `income_per_family_member` | Numerical *(Engineered)* | Per-capita income calculated as `annualincome / familymembers`. |
| `high_risk_traveler` | Binary *(Engineered)* | Combined indicator for frequent flyers with international travel history. |
| `is_high_income` | Binary *(Engineered)* | Flag indicating whether income meets or exceeds the dataset median. |
| `target` | Binary *(Target)* | Insurance purchase outcome (`1` = Purchased, `0` = Not Purchased). |

---

## 🏗️ Project Architecture & Workflow

1. **Data Ingestion & Cleaning (`step_1_load_and_clean_data`):**
   * Parses `Travel_Insurance_Prediction.csv`.
   * Standardizes column names to lowercase `snake_case`.
   * Standardizes the target variable column name to `target`.
   * Drops 738 duplicate entries, reducing the dataset from 1,987 to 1,249 clean observations.

2. **Advanced Feature Engineering (`step_2_feature_engineering`):**
   * **Per-Capita Income:** Captures household financial flexibility by dividing annual income by family size.
   * **High-Risk Traveler Flag:** Combines `frequentflyer` and `evertravelledabroad` into a single binary risk category.
   * **Income Threshold Indicator:** Flags affluent customers based on the median annual income benchmark.

3. **Exploratory Data Analysis (`step_3_exploratory_data_analysis`):**
   * Plots single-variable and bi-variable charts for all 12 dataset features against insurance purchase rates.

4. **Preprocessing & Modeling Pipeline:**
   * Implements `ColumnTransformer` and `Pipeline` architectures incorporating `SimpleImputer`, `StandardScaler` for numeric variables, and `OneHotEncoder` for categorical inputs.
   * Utilizes `StratifiedKFold` cross-validation to maintain class balance during evaluation.
   * Benchmark classifiers include `LogisticRegression`, `RandomForestClassifier`, and `XGBClassifier`.

5. **Inference & Model Performance Metrics:**
   * Evaluates predictions using Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and ROC-AUC curves (`roc_auc_score`).
