from fastapi import FastAPI
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from config import config
import uvicorn

app = FastAPI()

# Global variable for input features
input_features = []

# Load the trained model
def load_model():
    """Loads the trained model parameters from file."""
    model_path = os.path.join(config.SAVED_MODEL_PATH, config.SAVED_MODEL_FILE)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, "rb") as file:
        theta0, theta = pickle.load(file)
    
    return theta0, theta

# Sigmoid function
def sigmoid(theta0, theta, X):
    return 1 / (1 + np.exp(-(theta0 + np.matmul(X, theta))))

# Load scaler and extract feature names from REFINED_COLUMNS
def load_scaler():
    global input_features
    train_data_path = os.path.join(config.DATA_DIR, config.TRAINING_DATA_FILENAME)
    
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    train_data = pd.read_csv(train_data_path)

    # Remove target column to get input features
    input_features = [col for col in config.REFINED_COLUMNS if col != config.TARGET_COLUMN]
    train_features = train_data[input_features]

    scaler = StandardScaler()
    scaler.fit(train_features)
    
    return scaler

# Load model & scaler at startup
theta0, theta = load_model()
scaler = load_scaler()

# Input schema
class TumorInput(BaseModel):
    features: list[float]

# Root endpoint
@app.get("/")
async def home():
    return {"message": "Tumor Classification API is running!"}

# Features endpoint
@app.get("/features")
async def get_features():
    return {"features": input_features}

# Prediction endpoint
@app.post("/predict")
async def predict(input_data: TumorInput):
    if len(input_data.features) != len(input_features):
        return {
            "error": f"Expected {len(input_features)} features, but got {len(input_data.features)}"
        }

    input_array = np.array(input_data.features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probability = sigmoid(theta0, theta, input_scaled)
    prediction = int(probability > 0.5)

    return {
        "prediction": "Malignant" if prediction == 1 else "Benign",
        "probability": round(float(probability[0][0]), 4)
    }

# Run app locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
