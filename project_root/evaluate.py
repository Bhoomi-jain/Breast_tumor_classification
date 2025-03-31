import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from config import config

def load_data():
    """Load the test dataset."""
    test_data_path = os.path.join(config.DATA_DIR, config.TESTING_DATA_FILENAME)
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test dataset not found at {test_data_path}")
    
    test_data = pd.read_csv(test_data_path)
    
    return test_data

def preprocess_data(test_data):
    """Prepare the test dataset for evaluation."""
    X_test = np.array(test_data.iloc[:, 1:])  # Features
    y_test = np.array(test_data.iloc[:, 0])  # Labels
    
    y_test = y_test.reshape(y_test.shape[0], 1)
    
    # Standardize features (same transformation used in training)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    
    return X_test, y_test

def load_model():
    """Load the trained model from disk."""
    model_path = os.path.join(config.SAVED_MODEL_PATH, config.SAVED_MODEL_FILE)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    
    with open(model_path, "rb") as file:
        theta0, theta = pickle.load(file)
    
    return theta0, theta

def sigmoid(theta0, theta, X):
    """Compute the sigmoid function for prediction."""
    return 1 / (1 + np.exp(-(theta0 + np.matmul(X, theta))))

def evaluate():
    """Evaluate the trained model on the test dataset."""
    print("Loading test data...")
    test_data = load_data()
    
    print("Preprocessing test data...")
    X_test, y_test = preprocess_data(test_data)
    
    print("Loading trained model...")
    theta0, theta = load_model()
    
    print("Making predictions...")
    y_pred_prob = sigmoid(theta0, theta, X_test)
    y_pred = np.uint8(y_pred_prob > 0.5)  # Convert probabilities to binary predictions
    
    print("Computing evaluation metrics...")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy

if __name__ == "__main__":
    evaluate()
