import streamlit as st
import numpy as np
import pickle

# Set page title and layout
st.set_page_config(page_title="Diabetes Detection App", layout="wide")

# Title and Header
st.title("Diabetes Prediction App")
st.subheader("Enter your details to predict if you have diabetes.")

# Add a description
st.markdown("""
    This app predicts whether you have diabetes based on certain medical attributes.
    No need to upload a file! Simply fill in the details below and get your result instantly.
""")

# Input Fields: Collecting data from users
age = st.number_input("Age", min_value=18, max_value=100, value=30)
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=500, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
family_history = st.radio("Do you have a family history of diabetes?", ("Yes", "No"))

# Convert family history to numerical value for model
family_history_value = 1 if family_history == "Yes" else 0

# Button to submit
if st.button("Predict Diabetes"):
    # Process input and make prediction
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age, family_history_value])
    input_data = input_data.reshape(1, -1)

    # Load the trained model (make sure you have the pickle model saved)
    try:
        model = pickle.load(open("diabetes_model.pkl", "rb"))
        # Prediction
        prediction = model.predict(input_data)
        
        # Show the result
        if prediction[0] == 1:
            st.write("## Prediction: Diabetes Risk Detected")
        else:
            st.write("## Prediction: Low Risk of Diabetes")
    except Exception as e:
        st.write(f"Error loading model: {e}")

# Footer Information
st.markdown("""
    #### About
    This app is designed to help predict the likelihood of diabetes based on medical data. The model is built using machine learning techniques and trained on historical health data.
""")

# Option to add more information or links
st.markdown("[Learn more about diabetes](https://www.cdc.gov/diabetes/)")

