# ❤️ Heart Disease Prediction using Machine Learning

A machine learning project that predicts the presence or absence of heart disease using patient health-related features from the UCI Heart Disease Dataset.

🚀 **Live Demo:** https://jaisingh1414-disease-prediction-ml-app-so6mjm.streamlit.app/

## 📌 Project Overview

This project demonstrates a complete end-to-end machine learning workflow:

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
- Streamlit web application
- Cloud deployment

## 📊 Dataset

The project uses the **UCI Heart Disease Dataset**, specifically the processed Cleveland dataset.

The dataset contains:

- 303 patient records
- 13 input features
- 1 target variable

The original target contains values from `0` to `4`.

For this project, the target was converted into a binary classification problem:

| Target | Meaning |
|---|---|
| 0 | No heart disease |
| 1 | Heart disease |

## 🧾 Features

The model uses the following 13 features:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex of the patient |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |

## ⚙️ Data Preprocessing

The following preprocessing techniques were used:

- Missing values handled using imputation
- Numerical features standardized using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- `ColumnTransformer` used to combine preprocessing steps
- Preprocessing performed inside a machine learning pipeline
- Stratified train-test split used to preserve class distribution

The data was split into:

- **80% training data**
- **20% testing data**

## 🤖 Machine Learning Models

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

The initial Logistic Regression model provided the strongest overall test performance.

## 🔄 Cross-Validation

5-fold stratified cross-validation was performed to evaluate model stability.

| Model | Mean CV Accuracy |
|---|---:|
| Logistic Regression | 84.71% |
| Decision Tree | 69.80% |
| Random Forest | 81.81% |

## 🎯 Hyperparameter Tuning

`GridSearchCV` was used to tune the machine learning models.

### Logistic Regression

The best parameter found was:

```text
C = 10
```

Best cross-validation accuracy:

```text
85.54%
```

### Random Forest

The best parameters were:

```text
n_estimators = 100
max_depth = None
min_samples_split = 5
min_samples_leaf = 2
```

Best cross-validation accuracy:

```text
82.21%
```

## 🏆 Final Model

The tuned Logistic Regression pipeline was selected as the final model.

Final test performance:

| Metric | Score |
|---|---:|
| Test Accuracy | 86.89% |
| ROC-AUC | 96.23% |

The trained model was saved using Joblib:

```text
models/heart_disease_model.pkl
```

## 🔍 Prediction Example

A sample patient was provided to the trained model:

```text
Age: 55
Sex: Male
Chest Pain Type: 4
Resting Blood Pressure: 140
Cholesterol: 250
Fasting Blood Sugar: No
Resting ECG: 1
Maximum Heart Rate: 150
Exercise-Induced Angina: No
Oldpeak: 1.0
Slope: 2
Number of Major Vessels: 0
Thalassemia: 3
```

The model predicted:

```text
Prediction: Heart disease detected
Model probability: 56.56%
```

The probability shown by the application is the model's predicted probability and should not be interpreted as a clinically validated medical risk percentage.

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application where users can enter patient information and receive a machine learning prediction.

### 🚀 Live Demo

https://jaisingh1414-disease-prediction-ml-app-so6mjm.streamlit.app/

The application provides:

- Interactive patient input form
- Machine learning prediction
- Probability score
- User-friendly interface
- Educational disclaimer

## 📁 Project Structure

```text
disease-prediction-ml/
│
├── app.py
├── data/
│   ├── heart-disease.names
│   └── processed.cleveland.data
│
├── models/
│   └── heart_disease_model.pkl
│
├── notebooks/
│   └── disease_prediction.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook
- Streamlit
- Git & GitHub

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/jaisingh1414/disease-prediction-ml.git
```

### 2. Navigate to the project

```bash
cd disease-prediction-ml
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

On Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📚 Learning Outcomes

Through this project, I practiced:

- Data preprocessing
- Handling missing values
- Feature scaling
- Categorical feature encoding
- Machine learning pipelines
- Classification algorithms
- Model evaluation
- Confusion matrices
- ROC-AUC analysis
- Cross-validation
- Hyperparameter tuning
- Feature interpretation
- Model persistence using Joblib
- Building a Streamlit application
- Deploying an ML application to the cloud
- Using Git and GitHub for version control

## ⚠️ Disclaimer

This project is created for **educational and portfolio purposes only**.

It is not intended to provide medical diagnosis, treatment, or professional medical advice.

The predictions generated by this application should not be used for real-world medical decisions.

The deployed Streamlit application allows users to enter patient information and generate a heart disease prediction using the trained machine learning model.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit
## 👨‍💻 Author

**Jai Singh**

GitHub: https://github.com/jaisingh1414