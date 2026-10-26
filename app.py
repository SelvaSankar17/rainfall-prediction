    
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Rainfall Prediction",
    page_icon="🌧️",
    layout="centered"
)

# Load the saved model
MODEL_PATH = Path(__file__).parent / "rainfall_prediction_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model_data = pickle.load(file)

    model = model_data["model"]
    feature_names = model_data["feature_names"]

    return model, feature_names


st.title("🌧️ Rainfall Prediction")
st.write(
    "Enter the weather measurements below to predict "
    "whether rainfall is expected according to the trained model."
)

try:
    model, feature_names = load_model()

    # Input section
    st.subheader("Weather Details")

    col1, col2 = st.columns(2)

    with col1:
        pressure = st.number_input(
            "Pressure", value=1015.9, format="%.2f"
        )
        dewpoint = st.number_input(
            "Dew Point", value=19.9, format="%.2f"
        )
        humidity = st.number_input(
            "Humidity", min_value=0, max_value=100, value=95
        )
        cloud = st.number_input(
            "Cloud", min_value=0, max_value=100, value=81
        )

    with col2:
        sunshine = st.number_input(
            "Sunshine", min_value=0.0, value=0.0, format="%.2f"
        )
        winddirection = st.number_input(
            "Wind Direction", min_value=0.0,
            max_value=360.0, value=40.0, format="%.2f"
        )
        windspeed = st.number_input(
            "Wind Speed", min_value=0.0,
            value=13.7, format="%.2f"
        )

    # Predict rainfall
    if st.button("Predict Rainfall", type="primary"):
        input_values = {
            "pressure": pressure,
            "dewpoint": dewpoint,
            "humidity": humidity,
            "cloud": cloud,
            "sunshine": sunshine,
            "winddirection": winddirection,
            "windspeed": windspeed
        }

        input_df = pd.DataFrame(
            [[input_values[name] for name in feature_names]],
            columns=feature_names
        )

        prediction = model.predict(input_df)

        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.success("🌧️ Prediction: Rainfall")
        else:
            st.info("☀️ Prediction: No Rainfall")

        st.caption(
            "This is a machine-learning prediction for educational "
            "purposes, not an official weather forecast."
        )

except FileNotFoundError:
    st.error(
        "Model file not found. Keep rainfall_prediction_model.pkl "
        "in the same folder as app.py."
    )

except Exception as error:
    st.error(f"Unable to load or run the model: {error}")
