import numpy as np
import pandas as pd
import joblib
import os

#  Build the absolute path to the root of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#  Paths for scaler and model
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")  # conservé si utilisé ailleurs

COLUMNS = [ 
    "stops", "days_left", "duration", "class",
    "airline_Air_India", "airline_GO_FIRST", "airline_Indigo", "airline_SpiceJet", "airline_Vistara",
    "source_city_Chennai", "source_city_Delhi", "source_city_Hyderabad", "source_city_Kolkata", "source_city_Mumbai",
    "departure_time_Early_Morning", "departure_time_Evening", "departure_time_Late_Night", "departure_time_Morning", "departure_time_Night",
    "arrival_time_Early_Morning", "arrival_time_Evening", "arrival_time_Late_Night", "arrival_time_Morning", "arrival_time_Night",
    "destination_city_Chennai", "destination_city_Delhi", "destination_city_Hyderabad", "destination_city_Kolkata", "destination_city_Mumbai"
]

# Lazy-loaded scaler (pour faciliter les tests via monkeypatch)
_scaler = None
def _get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _scaler


def preprocess_user_input(data: dict) -> pd.DataFrame:
    """
    Transforme un dictionnaire d'entrée utilisateur en DataFrame prêt pour le modèle :
    - initialise toutes les features à 0
    - applique le scaling sur les features numériques
    - encode la classe (Business=1, sinon 0)
    - active les one-hot correspondants s'ils existent dans COLUMNS
    """
    # Put everything at 0
    input_df = pd.DataFrame([[0.0] * len(COLUMNS)], columns=COLUMNS, dtype=float)

    # Normalize the data
    numeric_features = ["stops", "days_left", "duration"]
    numeric_values = pd.DataFrame(
        [[data["stops"], data["days_left"], data["duration"]]],
        columns=numeric_features
    )

    scaler = _get_scaler()
    scaled_values = scaler.transform(numeric_values)
    input_df.loc[0, numeric_features] = scaled_values[0]

    # Classe : Business -> 1, sinon 0
    input_df.loc[0, "class"] = 1.0 if str(data["class_type"]).lower() == "business" else 0.0

    # One-hot activations
    one_hot_fields = ["airline", "source_city", "departure_time", "arrival_time", "destination_city"]
    for field in one_hot_fields:
        value = data[field]
        col_name = f"{field}_{value}"
        if col_name in input_df.columns:
            input_df.loc[0, col_name] = 1.0

    return input_df
