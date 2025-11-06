# app.py (summarized)
import os, traceback, pickle, joblib
import numpy as np
import streamlit as st

MODEL_PATH = "25RP18587.sav"  # change if needed

st.title("🌾 Crop Yield Prediction")
st.write("Enter temperature (°C) and click Predict.")

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"Missing package when unpickling: {e}")
    st.stop()
except Exception:
    st.error("Failed loading model — see traceback below.")
    st.text(traceback.format_exc())
    st.stop()

temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=100.0, step=0.1, value=27.0)

if st.button("Predict"):
    try:
        X = np.array([[float(temperature)]])
        pred = model.predict(X)
        st.success(f"🌱 Predicted Crop Yield: {float(pred[0]):.2f} units")
    except Exception:
        st.error("Prediction error:")
        st.text(traceback.format_exc())
