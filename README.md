## ✈️ Flight Ticket Price Prediction

This project aims to build a machine learning model capable of predicting the price of a flight ticket based on various features (such as airline, route, time, etc.).  
The final product is a simple web service that uses a REST API to receive input data and return predicted prices.

##  Technologies Used

- [Poetry](https://python-poetry.org/) — for dependency and virtual environment management  
- [DVC](https://dvc.org/) — for dataset and model versioning  
- [Pytest](https://docs.pytest.org/) — for unit testing  
- [FastAPI](https://fastapi.tiangolo.com/) — to expose the model as a REST API  
- Jupyter Notebook — for initial data exploration (EDA)

##  Project Goals

- Train a regression model to predict flight prices using tabular data  
- Serve the trained model through a RESTful API using FastAPI  
- Apply MLOps best practices: modular code, version control, data tracking, and automated testing  
- Build a codebase that is extensible and ready for production deployment  

##  Project Structure

```
flight-price-prediction/
├── data/           # Datasets (raw and processed), tracked with DVC
├── notebooks/      # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── ml/         # ML pipeline: preprocessing, training, inference
│   ├── api/        # FastAPI application
│   └── tests/      # Unit tests
├── dvc.yaml        # DVC pipeline configuration
├── pyproject.toml  # Poetry configuration file
├── README.md       # Project documentation
└── .gitignore
```

##  Getting Started

### 1. Clone the repository


git clone https://github.com/YOUR_USERNAME/flight-price-prediction.git
cd flight-price-prediction


### 2. Install dependencies with Poetry


poetry install
poetry shell


### 3. Set up DVC and download the dataset


dvc pull


##  Dataset

**Source**: *Flight Price Prediction - Kaggle*  
The dataset includes details such as airline, source and destination cities, number of stops, duration, and ticket prices.

##  Run the API and Front-End

Once the model (`random_forest_model.pkl`) and the scaler (`scaler.pkl`) are saved in the `models/` folder, you can run the full web service:

###  1. Start the FastAPI backend

In the root of the project:


poetry run uvicorn src.api.api:app --reload


If successful, you should see:


Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)


You can test the API at:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)

###  2. Start the Streamlit front-end

Open **another terminal** (keep the API running):


poetry run streamlit run app_front.py


Open the link shown in the terminal, e.g.:

[http://localhost:8501](http://localhost:8501)

### 3. Make a prediction

Fill in the form (choose airline, stops, departure time, etc.), then click **"Predict Price"**.
The API will return the estimated ticket price in real time.
