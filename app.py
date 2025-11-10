import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# 1️⃣ Page configuration
# -------------------------------------------------
st.set_page_config(page_title="BMW Price Prediction App", page_icon="🚗", layout="centered")
st.title("🚗 BMW Car Price Prediction App")
st.write("### Predict the price of a BMW car based on its specifications")

# -------------------------------------------------
# 2️⃣ Load model and scaler
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# -------------------------------------------------
# 3️⃣ Sidebar for input
# -------------------------------------------------
st.sidebar.header("Enter Car Details")

Year = st.sidebar.number_input("Year of Manufacture", 1990, 2025, 2020)
Age = st.sidebar.number_input("Car Age (in years)", 0, 30, 5)
Mileage_KM = st.sidebar.number_input("Mileage (in KM)", 0, 300000, 50000)
Price_per_KM = st.sidebar.number_input("Price per KM (approx)", 0.0, 100.0, 1.5)

Region = st.sidebar.selectbox("Region", ["Europe", "North America", "South America", "Asia"])
Fuel_Type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
Transmission = st.sidebar.selectbox("Transmission", ["Automatic", "Manual"])
Color = st.sidebar.selectbox("Color", ["White", "Black", "Silver", "Red", "Blue", "Grey"])
Model = st.sidebar.selectbox("Model", ["X1", "X3", "X5", "M3", "M5", "i3", "5 Series", "3 Series"])

# -------------------------------------------------
# 4️⃣ Convert input to DataFrame
# -------------------------------------------------
input_dict = {
    'Price_per_KM': [Price_per_KM],
    'Year': [Year],
    'Age': [Age],
    'Mileage_KM': [Mileage_KM],
    'Region': [Region],
    'Fuel_Type': [Fuel_Type],
    'Transmission': [Transmission],
    'Color': [Color],
    'Model': [Model]
}
input_data = pd.DataFrame(input_dict)

# -------------------------------------------------
# 5️⃣ One-hot encode and align columns to model’s features
# -------------------------------------------------
input_encoded = pd.get_dummies(input_data, drop_first=False)

# Get model’s trained feature names
model_features = model.get_booster().feature_names

# Add any missing columns from training
for col in model_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Remove any extra columns not in training
input_encoded = input_encoded[model_features]

# -------------------------------------------------
# 6️⃣ Scale using the scaler (trained on encoded features)
# -------------------------------------------------
input_scaled = scaler.transform(input_encoded)
input_scaled = pd.DataFrame(input_scaled, columns=model_features)

# -------------------------------------------------
# 7️⃣ Predict
# -------------------------------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)
    predicted_price = np.round(prediction[0], 2)
    st.success(f"💰 **Predicted Price (USD): ${predicted_price}**")

# -------------------------------------------------
# 8️⃣ Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and XGBoost")
