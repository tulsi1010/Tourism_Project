import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path_hf = hf_hub_download(repo_id="Tulsi10/Tourism", filename="best_model.joblib", repo_type="model")
best_model = joblib.load(model_path_hf)

# Streamlit app
st.title("Tourism Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Type = st.selectbox("Type", ["L0", "L1", "L2", "L3"])
Air_temperature = st.number_input("Air Temperature", min_value=-100.0, max_value=100.0, value=0.0)
Process_temperature = st.number_input("Process Temperature", min_value=-100.0, max_value=100.0, value=0.0)
Rotational_speed = st.number_input("Rotational Speed", min_value=0.0, max_value=10000.0, value=0.0)
Torque = st.number_input("Torque", min_value=0.0, max_value=10000.0, value=0.0)
Tool_wear = st.number_input("Tool Wear", min_value=0.0, max_value=100.0, value=0.0)

# Assemble input into DataFrame
input_data = pd.DataFrame({
    "Type": [Type],
    "Air temperature [K]": [Air_temperature],
    "Process temperature [K]": [Process_temperature],
    "Rotational speed [rpm]": [Rotational_speed],
    "Torque [Nm]": [Torque],
    "Tool wear [min]": [Tool_wear]
})

if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**") 
