import numpy as np
import pandas as pd
from src.ml.model import train_model

def test_train_model_returns_fitted_model():
    # mini dataset
    X = pd.DataFrame({
        "f1": [0, 1, 2, 3],
        "f2": [3, 2, 1, 0],
    })
    y = np.array([100, 120, 140, 160])

    model = train_model(X, y)

    assert hasattr(model, "predict")
    preds = model.predict(X)
    assert len(preds) == len(X)
