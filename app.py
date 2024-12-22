import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model and scaler
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set up the Streamlit app title and description
st.set_page_config(page_title="Diabetes Prediction App", page_icon=":guardsman:", layout="centered")
st.title("Diabetes Prediction")

st.markdown("""
Welcome to the **Diabetes Detection App**! 

This app predicts whether a person is likely to have diabetes based on a set of health metrics. Please input the required values and click on **Predict** to get your result.
""")

# Create input fields for user input with tooltips for better user guidance
age = st.number_input("Age (in years)", min_value=0, step=1, help="Enter your age in years.")
glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, step=1, help="Enter your glucose level (mg/dL).")
blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, step=1, help="Enter your blood pressure (mm Hg).")
skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, step=1, help="Enter the thickness of your skin fold (mm).")
insulin = st.number_input("Insulin Level (mu U/mL)", min_value=0, step=1, help="Enter your insulin level (mu U/mL).")
bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, step=0.1, help="Enter your Body Mass Index (BMI).")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1, help="Enter your Diabetes Pedigree Function value.")
pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, help="Enter the number of pregnancies you had.")

# Create a button to trigger prediction
if st.button("Predict"):
    # Organize the inputs into a dataframe for prediction
    user_input = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Scale the input data using the pre-fitted scaler
    user_input_scaled = scaler.transform(user_input)  # Use the pre-fitted scaler
    
    # Make prediction
    prediction = model.predict(user_input_scaled)

    # Display the result with improved styling
    st.subheader("Prediction Result:")

    if prediction[0] == 1:
        st.markdown("""
            <h2 style='color:red;'>The person is likely to have **Diabetes**.</h2>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <h2 style='color:green;'>The person is likely **not** to have **Diabetes**.</h2>
        """, unsafe_allow_html=True)

    # Display additional useful information
    st.markdown("""
    ---
    ### How this works:
    This app uses a **Support Vector Machine (SVM)** classifier trained on a dataset of medical and lifestyle factors. The classifier has learned to predict the likelihood of diabetes based on these factors.
    """)

# Footer section for additional info
st.markdown("""
---
*Made with ❤️ by [HARSH,SANJEEVANI,TRAPTI,CHAHAT]
""")
