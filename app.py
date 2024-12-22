import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Set page title and layout
st.set_page_config(page_title="Diabetes Prediction", layout="wide")

# Inject custom CSS for styling
st.markdown("""
    <style>
        /* Main background */
        body {
            background-color: #f1f1f1;
        }

        /* Customizing the title */
        .css-1e1w0bq {
            font-family: 'Arial', sans-serif;
            font-size: 36px;
            color: #4CAF50;
            text-align: center;
            padding-top: 20px;
        }

        /* Customizing description text */
        .css-1xv4n4o {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            color: #555;
            line-height: 1.6;
        }

        /* Customizing the form inputs */
        .streamlit-expanderHeader {
            font-size: 22px;
            font-weight: bold;
            color: #333;
        }

        .stSlider > div {
            font-size: 18px;
        }

        /* Submit button styles */
        .css-1y5x5t8 {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 5px;
            margin-top: 20px;
            transition: background-color 0.3s ease;
        }
        .css-1y5x5t8:hover {
            background-color: #45a049;
        }

        /* Prediction result styling */
        .stSuccess {
            font-size: 24px;
            font-weight: bold;
            color: #388e3c;
            background-color: #dcedc8;
            padding: 10px;
            border-radius: 5px;
        }

        /* Footer text styling */
        .footer {
            font-size: 14px;
            color: #777;
            text-align: center;
            margin-top: 30px;
            padding-bottom: 20px;
        }

    </style>
""", unsafe_allow_html=True)

# Add a header and some description
st.title("Diabetes Prediction App")
st.write("""
    Welcome to the Diabetes Prediction App! This tool will help you predict whether you are at risk of diabetes
    based on certain health parameters. Enter your details below and get your prediction.
""")

# Create a form to take input values
with st.form(key='diabetes_form'):
    # Input fields for the model (using slider for continuous values and number inputs)
    age = st.slider("Age", 0, 120, 25)
    bmi = st.slider("BMI (Body Mass Index)", 10, 50, 25)
    blood_pressure = st.slider("Blood Pressure (mm Hg)", 60, 200, 80)
    glucose = st.slider("Glucose Level", 50, 200, 100)
    insulin = st.slider("Insulin Level (mu U/mL)", 0, 500, 50)
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
    diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    pregnancies = st.slider("Number of Pregnancies", 0, 15, 1)

    # Add a submit button
    submit_button = st.form_submit_button(label='Predict')

# Define prediction logic after button is clicked
if submit_button:
    # Prepare the input data for prediction
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]])

    # Get the prediction
    prediction = model.predict(input_data)

    # Display the result with a message
    if prediction[0] == 1:
        st.success("Prediction: **You are at risk of diabetes**")
    else:
        st.success("Prediction: **You are not at risk of diabetes**")

# Add a footer with more information
st.markdown("""
    <div class="footer">
        **About the App:**
        This app uses machine learning to predict the likelihood of diabetes based on various health metrics. The model
        has been trained on a dataset and can provide insights on whether you should consult a doctor for further tests.
    </div>
""", unsafe_allow_html=True)
