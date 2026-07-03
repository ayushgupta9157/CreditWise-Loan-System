# CreditWise — Loan Approval Prediction System

An end-to-end supervised machine learning pipeline that predicts loan approval status using applicant data. The project covers exploratory data analysis, feature engineering, model training, and comparative evaluation across three classification algorithms.

## Problem Statement

Loan approval is framed as a **binary classification problem** — given applicant details (income, credit history, employment, etc.), predict whether a loan will be **approved** or **rejected**.

## Dataset

- File: `loan_approval_data.csv`
- **1,000 rows, 19 original features** (income, credit score, DTI ratio, savings, collateral, employment status, education, gender, property area, etc.)
- Target variable: `Loan_Approved` (binary) — class distribution: **722 Not Approved vs 278 Approved** (imbalanced)
- No missing values in the raw dataset
- Preprocessing: label/one-hot encoding of categorical features (expanded to 27 features post-encoding), feature scaling with `StandardScaler`
- Top correlated features with approval: **Credit_Score (+0.45)**, **DTI_Ratio (−0.44)**, **Loan_Amount (−0.13)**, **Applicant_Income (+0.12)**

## Approach

1. **Exploratory Data Analysis (EDA)** — understanding feature distributions, correlations, and class balance
2. **Feature Engineering** — preparing features for model input
3. **Model Training** — trained and compared three classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - Naive Bayes
4. **Evaluation** — Precision, Recall, F1-score, Accuracy, and Confusion Matrix for each model

## Results

### Before Feature Engineering

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| **Naive Bayes** | **0.8036** | 0.7377 | 0.7692 | 0.8650 |
| Logistic Regression | 0.7833 | 0.7705 | 0.7769 | 0.8650 |
| KNN | 0.6275 | 0.5246 | 0.5714 | 0.7600 |

At this stage, **Naive Bayes led on precision**, making it the initial choice for a precision-focused use case (minimizing false loan approvals).

### After Feature Engineering

| Model | Precision | Recall | F1 Score | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | **0.7903** | **0.8033** | **0.7967** | **0.8750** |
| Naive Bayes | 0.7833 | 0.7705 | 0.7769 | 0.8650 |
| KNN | 0.6200 | 0.5082 | 0.5586 | 0.7550 |

**Best Model: Logistic Regression** — after feature engineering (encoding categorical variables, scaling, and correlation-based feature selection), Logistic Regression overtook Naive Bayes on every metric, becoming the most balanced and reliable model for this task.

### Impact of Feature Engineering
Feature engineering improved Logistic Regression's precision from 0.7833 → 0.7903 and its accuracy from 0.865 → 0.875, flipping the best-model ranking from Naive Bayes to Logistic Regression. This highlights how encoding, scaling, and feature selection materially affect model performance — even when accuracy alone looks similar.

### Confusion Matrices

**Logistic Regression**
```
[[126  13]
 [ 12  49]]
```

**Naive Bayes**
```
[[126  13]
 [ 14  47]]
```

**KNN**
```
[[120  19]
 [ 30  31]]
```

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib / Seaborn (for EDA)
- Jupyter Notebook

## How to Run

```bash
git clone https://github.com/ayushgupta9157/CreditWise-Loan-System.git
cd CreditWise-Loan-System
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook credit_wise.ipynb
```

## Key Takeaways

- Logistic Regression outperformed both KNN and Naive Bayes across every evaluation metric, suggesting the decision boundary for this dataset is largely linear.
- KNN underperformed significantly, likely due to sensitivity to feature scaling and the curse of dimensionality on this dataset.
- Naive Bayes performed competitively despite its strong independence assumption between features.

## Author

**Ayush Gupta**
