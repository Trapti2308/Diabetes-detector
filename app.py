import streamlit as st
import pandas as pd
import pickle

# Load the pre-trained model
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the pre-fitted scaler (this is the solution to avoid the NotFittedError)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set up the Streamlit app
st.title("Diabetes Prediction")

# Create input fields for user input
age = st.number_input("Age", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.1)
pregnancies = st.number_input("Pregnancies", min_value=0)

# Create a button to trigger prediction
if st.button("Predict"):
    # Organize the inputs into a dataframe for prediction
    user_input = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]], 
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    
    # Scale the input data using the pre-fitted scaler
    user_input_scaled = scaler.transform(user_input)  # Use the pre-fitted scaler

    # Make prediction
    prediction = model.predict(user_input_scaled)

    # Display the result
    if prediction[0] == 1:
        st.write("Prediction: The person is likely to have Diabetes.")
    else:
        st.write("Prediction: The person is likely not to have Diabetes.")