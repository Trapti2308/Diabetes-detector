import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Set page title and layout
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Inject custom CSS for styling with animations and effects
st.markdown("""
    <style>
        /* Gradient Background */
        body {
            background: linear-gradient(45deg, #ff6f61, #ffcc5c, #6b8e23);
            background-size: 400% 400%;
            animation: gradientBG 5s ease infinite;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Animation for background gradient */
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Customizing the title */
        .css-1e1w0bq {
            font-family: 'Arial', sans-serif;
            font-size: 48px;
            color: #fff;
            text-align: center;
            margin-top: 50px;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
            font-weight: bold;
        }

        /* Customizing the description text */
        .css-1xv4n4o {
            font-family: 'Arial', sans-serif;
            font-size: 18px;
            color: #fff;
            line-height: 1.8;
            text-align: center;
        }

        /* Slider Styles */
        .stSlider {
            margin: 10px 0;
            width: 90%;
            background-color: #fff;
            border-radius: 10px;
            padding: 5px;
        }

        .stSlider > div {
            font-size: 18px;
            color: #333;
        }

        /* Custom Button Styles */
        .css-1y5x5t8 {
            background-color: #ff6f61;
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px 30px;
            border-radius: 5px;
            transition: transform 0.3s ease-in-out, background-color 0.3s ease;
            border: none;
        }

        .css-1y5x5t8:hover {
            background-color: #e94e4b;
            transform: scale(1.1);
        }

        /* Prediction Result Styles */
        .stSuccess {
            font-size: 24px;
            font-weight: bold;
            color: #388e3c;
            background-color: #dcedc8;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            animation: fadeIn 1s ease;
        }

        /* Emoji for Fun */
        .emoji {
            font-size: 30px;
            margin-left: 10px;
        }

        /* Footer Text */
        .footer {
            font-size: 14px;
            color: #fff;
            text-align: center;
            margin-top: 50px;
            padding-bottom: 30px;
        }

        /* Animation for showing results */
        @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
        }

    </style>
""", unsafe_allow_html=True)

# Add a header and description with emojis
st.title("🩺 **Diabetes Prediction App** 🩺")
st.write("""
    🌟 Welcome to the Diabetes Prediction App! 🌟
    Please provide your details below to predict your risk of diabetes. 
    Based on your health data, our model will make the prediction. 
    Let's get started! 👇
""")

# Create the form for taking input values
with st.form(key='diabetes_form'):
    st.subheader("🔢 Enter Your Health Details Below")
    
    # Input fields for model (using slider for continuous values and number inputs)
    age = st.slider("👵 Age", 0, 120, 25)
    bmi = st.slider("💪 BMI (Body Mass Index)", 10, 50, 25)
    blood_pressure = st.slider("💓 Blood Pressure (mm Hg)", 60, 200, 80)
    glucose = st.slider("🍭 Glucose Level", 50, 200, 100)
    insulin = st.slider("💉 Insulin Level (mu U/mL)", 0, 500, 50)
    skin_thickness = st.slider("🧴 Skin Thickness (mm)", 0, 100, 20)
    diabetes_pedigree_function = st.slider("👨‍👩‍👧‍👦 Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    pregnancies = st.slider("🤰 Number of Pregnancies", 0, 15, 1)

    # Submit button
    submit_button = st.form_submit_button(label='🔮 Predict')

# Logic to process prediction when the button is clicked
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

    # Display the result with animations and emojis
    if prediction[0] == 1:
        st.success(f"🚨 **Prediction: You are at risk of diabetes** 🚨")
    else:
        st.success(f"✅ **Prediction: You are not at risk of diabetes** ✅")

# Footer with additional info and emoji
st.markdown("""
    <div class="footer">
        💡 **About this App:** 💡
        This app uses machine learning to predict the likelihood of diabetes based on your health data.
        If you're at risk, please consult a healthcare professional for further testing. Stay healthy! 🌱
    </div>
""", unsafe_allow_html=True)
