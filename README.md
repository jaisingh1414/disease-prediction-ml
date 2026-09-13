# Heart Disease Prediction using Machine Learning

A machine learning project that predicts the presence or absence of heart disease using patient health-related features from the UCI Heart Disease dataset.

## Project Overview

This project demonstrates a complete machine learning workflow:

- Data collection and loading
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Train-test splitting
- Feature scaling and encoding
- Machine learning model training
- Model evaluation
- Cross-validation
- Hyperparameter tuning
- Feature interpretation
- ROC-AUC analysis
- Model saving and loading
- Prediction on new patient data

## Dataset

The project uses the **UCI Heart Disease Dataset**, specifically the processed Cleveland dataset.

The dataset contains:

- 303 patient records
- 13 input features
- 1 target variable

The original target contains values from `0` to `4`.

For this project, it was converted into a binary classification problem:

| Target | Meaning |
|---|---|
| 0 | No heart disease |
| 1 | Heart disease |

## Features

The model uses the following 13 features:

- Age
- Sex
- Chest pain type (`cp`)
- Resting blood pressure (`trestbps`)
- Cholesterol (`chol`)
- Fasting blood sugar (`fbs`)
- Resting ECG (`restecg`)
- Maximum heart rate (`thalach`)
- Exercise-induced angina (`exang`)
- ST depression (`oldpeak`)
- Slope (`slope`)
- Number of major vessels (`ca`)
- Thalassemia (`thal`)

## Data Preprocessing

The following preprocessing techniques were used:

- Missing values handled using imputation
- Numerical features standardized using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- `ColumnTransformer` used to combine numerical and categorical preprocessing
- Preprocessing was performed inside a machine learning pipeline to avoid data leakage

## Machine Learning Models

Three classification algorithms were evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

### Model Performance

| Model | Test Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 88.52% | ~96.65% |
| Decision Tree | 73.77% | ~74.40% |
| Random Forest | 86.89% | ~94.25% |

Logistic Regression provided the strongest overall performance on the initial test evaluation.

## Cross-Validation

5-fold stratified cross-validation was performed.

Initial mean cross-validation accuracy:

| Model | Mean CV Accuracy |
|---|---:|
| Logistic Regression | 84.71% |
| Decision Tree | 69.80% |
| Random Forest | 81.81% |

## Hyperparameter Tuning

GridSearchCV was used to improve model performance.

### Logistic Regression

Best parameter:

```text
C = 10