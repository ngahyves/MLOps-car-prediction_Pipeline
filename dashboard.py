# dashboard.py

# --- Step 1 : Import the libraries
import streamlit as st
import requests

# --- Step 2 : Design the interface
st.title("Car Price Estimator")

# Create user typing or selecting options
manufacturer = st.text_input("Manufacturer", value="ford")
model_reduced = st.text_input("Model", value="f-150")
age = st.number_input("Car Age (years)", value=10)
odometer = st.number_input("Odometer (miles)", value=150000)

# Create a drop down list
transmission = st.selectbox("Transmission", ["automatic", "manual", "other"])
fuel = st.selectbox("Fuel Type", ["gas", "diesel", "other"])
title_status = st.selectbox("Title Status", ["clean", "rebuilt", "salvage"])


# --- Step 3 : Create a button's action ---
if st.button("Predict Price"):
    
    # 1. Prepare the data to send to the API
    car_data = {
        "manufacturer": manufacturer,
        "model_reduced": model_reduced,
        "transmission": transmission,
        "fuel": fuel,
        "title_status": title_status,
        "age": age,
        "odometer": odometer
    }

    # 2. Send the data to the API and make a request
    response = requests.post("http://127.0.0.1:8000/predict", json=car_data)

    # 3. Gather the prediction
    prediction = response.json()
    
    # Prediction
    price = prediction['predicted_price']

    # Display the predicted price
    st.success(f"The estimated price of the car is ${price}")