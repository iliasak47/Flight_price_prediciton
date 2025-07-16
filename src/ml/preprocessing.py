import numpy as np
import pandas as pd
import joblib
import os


#  Build the absolute path to the root of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#  Paths for scaler and model
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")

#  Load the scaler and the trained model
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

COLUMNS = [ 
    "stops", "days_left", "duration", "class",
    "airline_Air_India", "airline_GO_FIRST", "airline_Indigo", "airline_SpiceJet", "airline_Vistara",
    "source_city_Chennai", "source_city_Delhi", "source_city_Hyderabad", "source_city_Kolkata", "source_city_Mumbai",
    "departure_time_Early_Morning", "departure_time_Evening", "departure_time_Late_Night", "departure_time_Morning", "departure_time_Night",
    "arrival_time_Early_Morning", "arrival_time_Evening", "arrival_time_Late_Night", "arrival_time_Morning", "arrival_time_Night",
    "destination_city_Chennai", "destination_city_Delhi", "destination_city_Hyderabad", "destination_city_Kolkata", "destination_city_Mumbai"
]

def preprocess_user_input(data: dict):

    # Put everything at 0
    input_df = pd.DataFrame([[0.0] * len(COLUMNS)], columns=COLUMNS, dtype=float)


    # Normalize the data
    numeric_features = ["stops", "days_left", "duration"]
    numeric_values = pd.DataFrame(
    [[data["stops"], data["days_left"], data["duration"]]],
    columns=numeric_features)
    scaled_values = scaler.transform(numeric_values)
    input_df.loc[0, numeric_features] = scaled_values[0]

    input_df.loc[0, "class"] = 1 if data["class_type"].lower() == "business" else 0
    # input_df.loc[0, "class"] = 1 if data["class"].lower() == "business" else 0

    # Make 1 if the column is present
    one_hot_fields = [
        "airline", "source_city", "departure_time", "arrival_time", "destination_city"
    ]

    for field in one_hot_fields:
        col_name = f"{field}_{data[field]}"
        if col_name in input_df.columns:
            input_df.loc[0, col_name] = 1

    return input_df

