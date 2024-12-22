import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model (diabetes_model.pkl)
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('/content/scaler.pkl', 'rb'))  # If you have used scaling for input features

# Set up the app title and header with some animation
st.title("Diabetes Detection 🚨")
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Predict Diabetes With Confidence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Fill out the details and get your prediction below 👇</p>", unsafe_allow_html=True)

# Add some style to the page
st.markdown("""
    <style>
        .stButton>button {
            background-color: #FF6347;
            color: white;
            font-size: 20px;
            height: 3em;
            border-radius: 10px;
        }
        .stTextInput input {
            background-color: #f7f7f7;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
        }
        .stTextArea textarea {
            background-color: #f7f7f7;
            border-radius: 8px;
            padding: 12px;
            font-size: 16px;
        }
        .stSlider .st-bx {
            font-size: 18px;
            color: #FF6347;
        }
        .stRadio label {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar with input fields for the user
st.sidebar.header("Input Parameters 📝")
age = st.sidebar.number_input("Age (Years):", min_value=1, max_value=100, value=25)
bmi = st.sidebar.slider("Body Mass Index (BMI):", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
glucose = st.sidebar.slider("Glucose Level (mg/dL):", min_value=50, max_value=200, value=95)
blood_pressure = st.sidebar.slider("Blood Pressure (mm Hg):", min_value=50, max_value=200, value=80)
skin_thickness = st.sidebar.slider("Skin Thickness (mm):", min_value=10, max_value=100, value=20)
insulin = st.sidebar.slider("Insulin (μU/mL):", min_value=10, max_value=800, value=100)
diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function:", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
pregnancies = st.sidebar.number_input("Number of Pregnancies:", min_value=0, max_value=10, value=0)

# Create a button to predict the result
button = st.sidebar.button("Predict Result 🤔")

# Main logic to process inputs and show prediction
if button:
    # Input features into a numpy array
    input_features = np.array([age, bmi, glucose, blood_pressure, skin_thickness, insulin, diabetes_pedigree, pregnancies]).reshape(1, -1)
    
    # Apply scaling if required (ensure you have fitted a scaler)
    scaled_input = scaler.transform(input_features)
    
    # Make prediction using the loaded model
    prediction = model.predict(scaled_input)
    
    # Display the result with emojis and a nice message
    if prediction == 1:
        st.markdown("<h2 style='text-align: center; color: #FF6347;'>⚠️ You are at risk of diabetes! ⚠️</h2>", unsafe_allow_html=True)
        st.image("https://example.com/diabetes_warning.jpg", caption="Warning: Diabetes Risk", use_column_width=True)
        st.markdown("<p style='text-align: center; font-size: 18px;'>Please consult with a doctor and take necessary actions 🩺</p>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='text-align: center; color: #28A745;'>✅ You are not at risk of diabetes! ✅</h2>", unsafe_allow_html=True)
        st.image("https://example.com/healthy_lifestyle.jpg", caption="Healthy Lifestyle", use_column_width=True)
        st.markdown("<p style='text-align: center; font-size: 18px;'>Keep up with a healthy diet and lifestyle 💪</p>", unsafe_allow_html=True)

# Adding more interactivity to the page using animations
st.markdown("""
    <style>
        .stMarkdown h2 {
            animation: fadein 2s ease-in-out;
        }
        @keyframes fadein {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)
