import importlib
import pandas as pd

def test_preprocess_shape_and_onehots(monkeypatch, fake_scaler_identity, sample_input_business):
    from src.ml import preprocessing as prep
    monkeypatch.setattr(prep, "_get_scaler", lambda: fake_scaler_identity)

    X = prep.preprocess_user_input(sample_input_business)

    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] == 1
    assert list(X.columns) == prep.COLUMNS  

    assert X.loc[0, "airline_Indigo"] == 1.0
    assert X.loc[0, "source_city_Delhi"] == 1.0
    assert X.loc[0, "departure_time_Morning"] == 1.0
    assert X.loc[0, "arrival_time_Evening"] == 1.0
    assert X.loc[0, "destination_city_Mumbai"] == 1.0

    assert X.loc[0, "class"] == 1.0  # Business => 1

def test_preprocess_numeric_scaling(monkeypatch, fake_scaler_scale2, sample_input_business):
    from src.ml import preprocessing as prep
    monkeypatch.setattr(prep, "_get_scaler", lambda: fake_scaler_scale2)

    X = prep.preprocess_user_input(sample_input_business)

    assert X.loc[0, "stops"] == sample_input_business["stops"] / 2
    assert X.loc[0, "days_left"] == sample_input_business["days_left"] / 2
    assert X.loc[0, "duration"] == sample_input_business["duration"] / 2
