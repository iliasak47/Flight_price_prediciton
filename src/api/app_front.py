import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000/predict"

st.title("Flight Price Prediction")
st.write("This app estimates the price of a flight ticket based on your trip details.")

# User input 
st.subheader("Enter your flight details")

stops = st.selectbox("Number of stops", [0, 1, 2])
days_left = st.slider("Days left before the flight", 1, 60, 15)
duration = st.slider("Flight duration (hours)", 1.0, 12.0, 4.5)
class_type = st.selectbox("Class", ["Economy", "Business"])

airline = st.selectbox("Airline", ["Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"])
source_city = st.selectbox("Source City", ["Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])
destination_city = st.selectbox("Destination City", ["Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"])

departure_time = st.selectbox("Departure Time", ["Early_Morning", "Evening", "Late_Night", "Morning", "Night"])
arrival_time = st.selectbox("Arrival Time", ["Early_Morning", "Evening", "Late_Night", "Morning", "Night"])

# --- Prediction button ---
if st.button("Predict Price"):
    data = {
        "stops": stops,
        "days_left": days_left,
        "duration": duration,
        "class_type": class_type,
        "airline": airline,
        "source_city": source_city,
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "destination_city": destination_city
    }

    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Estimated Price: ₹ {result['predicted_price']}")
    else:
        st.error("Error while fetching prediction. Check if the FastAPI server is running.")
