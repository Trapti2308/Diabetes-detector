import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
model = joblib.load('diabetes_model.pkl')  # Change this to your actual model path

# Streamlit UI (Title, Inputs, and Prediction Output)
st.title('Diabetes Prediction Web App')

# Create input fields for user
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0)
insulin = st.number_input('Insulin', min_value=0, max_value=1000, value=0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.0)
age = st.number_input('Age', min_value=0, max_value=120, value=0)

# Button for prediction
if st.button('Predict'):
    # Prepare input data as a DataFrame
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    if prediction == 0:
        st.success('Prediction: No Diabetes')
    else:
        st.success('Prediction: Diabetes')
