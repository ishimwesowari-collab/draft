import pickle
import numpy as np
import streamlit as st

# Load your trained model
with open("25RP18587.sav", "rb") as f:
    model = pickle.load(f)

# Streamlit app title and description
st.title("🌾 Crop Yield Prediction App")
st.write("Enter the temperature value to predict the crop yield.")

# User input for temperature
temperature = st.number_input(
    "Enter the temperature (°C)",
    min_value=0.0,
    step=0.1,
    value=27.0
)

# Prediction button
if st.button("Predict Yield"):
    # Prepare input data for prediction
    X = np.array([[float(temperature)]])  # 2D array for model
    pred = model.predict(X)

    # Display result
    st.success(f"🌱 Predicted Crop Yield: *{pred[0]:.2f}* units")