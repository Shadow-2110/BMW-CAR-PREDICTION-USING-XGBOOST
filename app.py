import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# -------------------------------------------------
# 1️⃣ Page configuration
# -------------------------------------------------
st.set_page_config(page_title="BMW Price Prediction App", page_icon="🚗", layout="centered")
st.title("🚗 BMW Car Price Prediction App")
st.write("Predict the car price in USD based on technical and regional features.")

# -------------------------------------------------
# 2️⃣ Load trained model
# -------------------------------------------------
# Make sure you have run this earlier in Jupyter:
# joblib.dump(model, "xgb_model.pkl")

model = joblib.load("xgb_model.pkl")   # Load your actual trained model

# -------------------------------------------------
# 3️⃣ Define input fields for user
# -------------------------------------------------
st.header("🧩 Input Car Details")

col1, col2 = st.columns(2)

with col1:
    Year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2020)
    Mileage_KM = st.number_input("Mileage (in KM)", min_value=0, max_value=300000, value=20000)
    Sales_Volume = st.number_input("Sales Volume", min_value=0, max_value=5000, value=500)

with col2:
    Transmission = st.selectbox("Transmission Type", ['Automatic', 'Manual'])
    Fuel_Type = st.selectbox("Fuel Type", ['Petrol', 'Hybrid', 'Electric'])
    Region = st.selectbox("Region", ['Europe', 'Asia', 'South America', 'Middle East', 'North America'])
    Color = st.selectbox("Color", ['Red', 'Silver', 'White', 'Grey', 'Blue'])
    Model = st.selectbox("Model", ['5 Series', '7 Series', 'M3', 'M5', 'X1', 'X3', 'X5', 'X6', 'i3', 'i8'])
    Sales_Classification = st.selectbox("Sales Classification", ['Low', 'Medium', 'High'])

# -------------------------------------------------
# 4️⃣ Feature engineering (must match training logic)
# -------------------------------------------------
Age = 2025 - Year
Price_per_KM = 1 / (Mileage_KM + 1)  # to mimic normalized price-per-km ratio

# -------------------------------------------------
# 5️⃣ Create dataframe for prediction
# -------------------------------------------------
input_data = pd.DataFrame({
    'Price_per_KM': [Price_per_KM],
    'Year': [Year],
    'Age': [Age],
    'Mileage_KM': [Mileage_KM],
    'Region_South America': [1 if Region == 'South America' else 0],
    'Fuel_Type_Hybrid': [1 if Fuel_Type == 'Hybrid' else 0],
    'Model_X3': [1 if Model == 'X3' else 0],
    'Transmission_Manual': [1 if Transmission == 'Manual' else 0],
    'Color_Silver': [1 if Color == 'Silver' else 0],
    'Fuel_Type_Petrol': [1 if Fuel_Type == 'Petrol' else 0],
    'Model_M5': [1 if Model == 'M5' else 0],
    'Model_X1': [1 if Model == 'X1' else 0],
    'Model_i3': [1 if Model == 'i3' else 0],
    'Color_Red': [1 if Color == 'Red' else 0],
    'Model_5 Series': [1 if Model == '5 Series' else 0]
})

# -------------------------------------------------
# 6️⃣ Scale numeric columns
# -------------------------------------------------
scaler = StandardScaler()
num_cols = ['Price_per_KM', 'Year', 'Age', 'Mileage_KM']
input_data[num_cols] = scaler.fit_transform(input_data[num_cols])

# all features used during training
expected_features = [
    'Year', 'Engine_Size_L', 'Mileage_KM', 'Sales_Volume', 'Age', 'Price_per_KM',
    'Engine_Power_Ratio', 'Volume_per_Price', 'Model_5 Series', 'Model_7 Series',
    'Model_M3', 'Model_M5', 'Model_X1', 'Model_X3', 'Model_X5', 'Model_X6',
    'Model_i3', 'Model_i8', 'Region_Asia', 'Region_Europe', 'Region_Middle East',
    'Region_North America', 'Region_South America', 'Color_Blue', 'Color_Grey',
    'Color_Red', 'Color_Silver', 'Color_White', 'Fuel_Type_Electric',
    'Fuel_Type_Hybrid', 'Fuel_Type_Petrol', 'Transmission_Manual',
    'Sales_Classification_Low'
]

# make sure all columns exist
for col in expected_features:
    if col not in input_data.columns:
        input_data[col] = 0

# reorder columns to match training order
input_data = input_data[expected_features]

# now predict safely
y_pred = model.predict(input_data)


# -------------------------------------------------
# 7️⃣ Predict price
# -------------------------------------------------
if st.button("💰 Predict Price"):
    y_pred = model.predict(input_data)
    predicted_price = float(y_pred[0])

    st.success(f"### 💵 Estimated BMW Price: ${predicted_price:,.2f}")
    st.balloons()

    st.subheader("🔍 Input Summary:")
    st.dataframe(input_data)

# -------------------------------------------------
# 8️⃣ Footer
# -------------------------------------------------
st.write("---")
st.caption("Developed by Samrat 🚀 | Powered by Streamlit & XGBoost")
