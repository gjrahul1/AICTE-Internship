import streamlit as st
import pickle
import numpy as np
import requests
import io

# GitHub base URL for raw files
github_base_url = "https://raw.githubusercontent.com/gjrahul1/AICTE-Internship/main/training_models/"

# Function to load model from GitHub
def load_model(filename):
    url = github_base_url + filename
    response = requests.get(url)
    response.raise_for_status()
    return pickle.load(io.BytesIO(response.content))

# Load models
diabetes_model = load_model("diabetes_model.sav")
heart_model = load_model("heart_model.sav")
parkinsons_model = load_model("parkinsons_model.sav")

# Page title
st.title("Health Prediction Application")

# Model selection
option = st.selectbox("Select a health prediction model:", ["Diabetes", "Heart Disease", "Parkinson's"])

# Input fields based on model selection
if option == "Diabetes":
    st.subheader("Diabetes Prediction")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    inputs = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    model = diabetes_model

elif option == "Heart Disease":
    st.subheader("Heart Disease Prediction")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=120)
    chol = st.number_input("Cholesterol Level", min_value=0, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved", min_value=0, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.number_input("Number of Major Vessels (0-4)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])
    inputs = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
    model = heart_model

else:
    st.subheader("Parkinson's Disease Prediction")
    fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=300.0, value=150.0)
    fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=500.0, value=200.0)
    flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=300.0, value=120.0)
    jitter_percent = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=1.0, value=0.01)
    shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, value=0.02)
    nhr = st.number_input("NHR", min_value=0.0, max_value=1.0, value=0.02)
    hnr = st.number_input("HNR", min_value=0.0, max_value=40.0, value=20.0)
    rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.4)
    dfa = st.number_input("DFA", min_value=0.0, max_value=2.0, value=0.6)
    spread1 = st.number_input("Spread1", min_value=-10.0, max_value=5.0, value=-4.0)
    spread2 = st.number_input("Spread2", min_value=0.0, max_value=1.0, value=0.2)
    d2 = st.number_input("D2", min_value=0.0, max_value=5.0, value=2.5)
    inputs = np.array([[fo, fhi, flo, jitter_percent, shimmer, nhr, hnr, rpde, dfa, spread1, spread2, d2]])
    model = parkinsons_model

# Predict button
if st.button("Predict"):
    prediction = model.predict(inputs)
    if prediction[0] == 1:
        st.error(f"The model predicts a positive case for {option}.")
    else:
        st.success(f"The model predicts a negative case for {option}.")
