import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Title and Description
st.set_page_config(page_title="Diabetes Detection Visualization", layout="wide")
st.title("🌟 Diabetes Detection Visualization Dashboard 🌟")
st.markdown(
    """
    **Analyze and visualize diabetes-related data using interactive and dynamic plots.**
    """
)

# Upload Dataset
st.sidebar.header("/content/diabetes.csv")
uploaded_file = st.sidebar.file_uploader("/content/diabetes.csv", type=["csv"])
if uploaded_file is not None:
    diabetes_data = pd.read_csv(uploaded_file)
    st.sidebar.success("File Uploaded Successfully!")

    # Display Dataset
    st.header("Dataset Preview")
    st.write(diabetes_data.head())

    # Feature Selection
    st.sidebar.header("Choose Features for Visualization")
    feature_columns = diabetes_data.columns.tolist()
    selected_features = st.sidebar.multiselect(
        "Select Features", feature_columns, default=feature_columns[:4]
    )

    # Distribution Plots
    st.subheader("📊 Feature Distributions")
    for feature in selected_features:
        st.write(f"**{feature} Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(diabetes_data[feature], kde=True, bins=30, color="blue", ax=ax)
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation = diabetes_data.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Pairplot for Selected Features
    st.subheader("📈 Pairplot of Features")
    if len(selected_features) > 1:
        sns.pairplot(diabetes_data[selected_features])
        st.pyplot()

    # Boxplot
    st.subheader("📊 Boxplot by Outcome")
    for feature in selected_features:
        if feature != "Outcome":
            st.write(f"**{feature} vs Outcome**")
            fig, ax = plt.subplots()
            sns.boxplot(x="Outcome", y=feature, data=diabetes_data, palette="Set3", ax=ax)
            st.pyplot(fig)
else:
    st.warning("Please upload a dataset to proceed.")

# Footer
st.markdown("### Created with 💖 using Streamlit")
