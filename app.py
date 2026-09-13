import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/heart_disease_model.pkl")

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient information below to get a machine learning prediction.")

st.subheader("Patient Information")

age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

restecg = st.selectbox(
    "Resting ECG Results",
    options=[0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise-Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[1, 2, 3]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    options=[3, 6, 7]
)

if st.button("Predict"):

    new_patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = model.predict(new_patient)[0]
    probability = model.predict_proba(new_patient)[0][1]

    if prediction == 1:
        st.error("Prediction: Heart disease detected")
    else:
        st.success("Prediction: No heart disease detected")

    st.write(f"Probability of heart disease: {probability:.2%}")