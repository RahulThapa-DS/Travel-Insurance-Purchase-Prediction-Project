# Travel-Insurance-Purchase-Prediction---Project
# Travel Insurance Purchase Prediction & Analytics System

An end-to-end Machine Learning and exploratory analytics pipeline built to predict customer travel insurance purchasing decisions using demographic and behavioral data.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Model Performance Summary](#model-performance-summary)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## About The Project

This project delivers a binary classification framework to predict whether a targeted traveler will opt for travel insurance. By combining automated data preprocessing, feature engineering, and robust model evaluation across seven distinct algorithm configurations, this repository helps optimize conversion rates and streamline targeted marketing campaigns.

> [!NOTE]
> The primary metric targeted during evaluation is the **F1-Score**, prioritizing a balanced tradeoff between Precision (minimizing false sales leads) and Recall (capturing actual buyers).

---

## Key Features

- **Automated Exploratory Data Analysis (EDA):** Generates distribution histograms with mean lines for continuous numerical features and exploded pie charts for categorical variables across 12 feature fields.
- **Robust Machine Learning Pipeline:** Automatically runs, evaluates, and ranks Gradient Boosting, XGBoost (Tuned & Baseline), AdaBoost, Extra Trees, Random Forest, and Logistic Regression models.
- **Detailed Evaluation Metrics:** Tracks Accuracy, Precision, Recall, F1-Score, and ROC-AUC along with full confusion matrix counts (TP, TN, FP, FN).
- **Inference Summaries:** Outputs real-time prediction distribution breakdowns for 250 evaluation test customers.

---

## Tech Stack

| Component | Technology / Library |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Data Visualization** | Seaborn, Matplotlib |
| **Environment** | JupyterLab / Jupyter Notebook |

---

## Getting Started

Follow these steps to set up and run the notebook environment locally.

### Prerequisites

Ensure you have the following installed:
* Python 3.9+
* `pip` package manager
* `git`

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RahulThapa-DS/Travel-Insurance-Purchase-Prediction.git](https://github.com/RahulThapa-DS/Travel-Insurance-Purchase-Prediction.git)
   cd Travel-Insurance-Purchase-Prediction
