import joblib
from model import train_model

# Load data
X_train = joblib.load("data/processed/X_train_final.pkl")
y_train = joblib.load("data/processed/y_train.pkl")

# Train model
model = train_model(X_train, y_train)

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

