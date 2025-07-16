from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from src.ml.preprocessing import preprocess_user_input

app = FastAPI(
    title="Flight Price Prediction API",
    description="This API predicts flight ticket prices based on user input.",
    version="1.0"
)


model = joblib.load("models/random_forest_model.pkl")

@app.get("/")
def home():
    """
    Check route to confirm the API is running
    """
    return {"message": "Welcome to the Flight Price Prediction API!"}


# structure for input
class FlightInput(BaseModel):
    stops: int
    days_left: int
    duration: float
    class_type: str           
    airline: str              
    source_city: str          
    departure_time: str       
    arrival_time: str         
    destination_city: str     


@app.post("/predict")
def predict_price(data: FlightInput):
    """
    This endpoint receives flight details,
    preprocesses them into the right format,
    and predicts the flight price using the trained model.
    """

    # Convert Pydantic model to a dictionary
    data_dict = data.dict()

    # Prepare the input for the model (use your preprocessing function)
    X_ready = preprocess_user_input(data_dict)

    # Make the prediction
    predicted_price = model.predict(X_ready)[0]

    return {
        "predicted_price": round(predicted_price, 2),
        "currency": "INR",
        "details": "The price is an estimation based on your inputs"
    }
