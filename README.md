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

## 🧠 Methodology, Dataset & EDA Summary

### 1. Methodology & Model Selection
* **Problem Formulation:** Binary classification predicting customer travel insurance purchase intent (`1` = Purchased, `0` = Not Purchased).
* **Validation Strategy:** `StratifiedKFold` cross-validation implemented within Scikit-Learn `Pipeline` architectures to prevent data leakage and preserve target class proportions across cross-validation splits and hyperparameter optimization (`GridSearchCV`).
* **Selected Model:** **Gradient Boosting Classifier** (Best Overall Performance).
* **Selection Rationale:**
  * **High Precision (84.00%) & F1-Score (57.14%):** Maximizes marketing ROI by minimizing False Positives (only 8 misclassified non-buyers out of 250 test cases).
  * **Non-Linear Interactions:** Effectively models complex interactions between engineered variables (`income_per_family_member`, `high_risk_traveler`) without requiring linear assumptions.
  * **Robustness:** Sequential boosting balances model bias and variance when properly constrained via hyperparameter tuning.

---

### 2. Dataset Architecture & Preprocessing
* **Dataset Size:** Reduced from 1,987 raw rows to 1,249 unique records after dropping 738 duplicate entries to eliminate evaluation bias.
* **Feature Transformation Pipeline (`ColumnTransformer`):**
  * **Numerical Pipeline:** Missing values imputed via `SimpleImputer(strategy='median')` and scaled using `StandardScaler`.
  * **Categorical Pipeline:** Missing values imputed via `SimpleImputer(strategy='most_frequent')` and encoded via `OneHotEncoder(handle_unknown='ignore')`.
* **Engineered Features:**
  * `income_per_family_member`: Per-capita income (`annualincome / familymembers`) measuring household financial flexibility.
  * `high_risk_traveler`: Combined indicator combining `frequentflyer` and `evertravelledabroad`.
  * `is_high_income`: Binary flag indicating whether income meets or exceeds the dataset median.

---

### 3. Exploratory Data Analysis & Anomaly Handling
* **Key EDA Drivers:** 
  * Annual income (`annualincome`) and per-capita income (`income_per_family_member`) serve as the primary predictors of purchase conversion.
  * Prior travel experience (`evertravelledabroad`) and frequent flyer status (`frequentflyer`) exhibit high correlation with insurance adoption.
* **Missing & Outlier Value Handling:**
  * **Missing Values:** Handled dynamically inside cross-validation splits via median imputation for continuous variables and mode imputation for categorical attributes.
  * **Outliers & Skewness:** Numerical features were standardized using `StandardScaler`. Ensemble tree-based classifiers (Gradient Boosting, XGBoost, Random Forest) were prioritized as they are naturally invariant to monotonic feature scaling and robust against heavy-tailed distributions.
