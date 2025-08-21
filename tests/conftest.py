import numpy as np
import pandas as pd
import pytest

class FakeScaler:
    def __init__(self, means, scales):
        self.means = np.array(means, dtype=float)
        self.scales = np.array(scales, dtype=float)

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        # (X - mean) / scale
        return (X - self.means) / self.scales

@pytest.fixture
def fake_scaler_identity():
    return FakeScaler(means=[0.0, 0.0, 0.0], scales=[1.0, 1.0, 1.0])

@pytest.fixture
def fake_scaler_scale2():
    # divided by 2
    return FakeScaler(means=[0.0, 0.0, 0.0], scales=[2.0, 2.0, 2.0])

@pytest.fixture
def sample_input_business():
    return {
        "stops": 1,
        "days_left": 10,
        "duration": 4.5,
        "class_type": "Business",
        "airline": "Indigo",
        "source_city": "Delhi",
        "departure_time": "Morning",
        "arrival_time": "Evening",
        "destination_city": "Mumbai",
    }
