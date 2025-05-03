from fastapi import FastAPI
import numpy as np
import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
from config import config

app = FastAPI()

# Load the trained model
def load_model():
    """Loads the trained model parameters from file."""
    model_path = os.path.join(config.SAVED_MODEL_PATH, config.SAVED_MODEL_FILE)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    with open(model_path, "rb") as file:
        theta0, theta = pickle.load(file)
    
    return theta0, theta

# Sigmoid function for probability calculation
def sigmoid(theta0, theta, X):
    return 1 / (1 + np.exp(-(theta0 + np.matmul(X, theta))))

# Load scaler (fitted on training data)
def load_scaler():
    train_data_path = os.path.join(config.DATA_DIR, config.TRAINING_DATA_FILENAME)
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    train_data = pd.read_csv(train_data_path)
    train_features = train_data.iloc[:, 1:]  # Exclude target column
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    return scaler

# Load model & scaler
theta0, theta = load_model()
scaler = load_scaler()

# Define the input data format
class TumorInput(BaseModel):
    features: list[float]  # Expecting a list of feature values

@app.get("/")
def home():
    return {"message": "Tumor Classification API is running!"}

@app.post("/predict")
def predict(input_data: TumorInput):
    """Predict tumor classification based on input features."""
    # Convert input to a NumPy array
    input_array = np.array(input_data.features).reshape(1, -1)

    # Scale input
    input_scaled = scaler.transform(input_array)

    # Make prediction
    probability = sigmoid(theta0, theta, input_scaled)
    prediction = int(probability > 0.5)  # Convert to 0 (Benign) or 1 (Malignant)

    return {
        "prediction": "Malignant" if prediction == 1 else "Benign",
        "probability": round(float(probability[0][0]), 4)
    }

# Run FastAPI when executing this script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
