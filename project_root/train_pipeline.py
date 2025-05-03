import os
import pickle
import numpy as np
import pandas as pd
from training.train import training
from transform_pipeline import complete_pipeline
from config import config

def run_training():
    print(" Running the data transformation pipeline...")

    # Step 1: Run Data Transformation Pipeline
    training_data, testing_data = complete_pipeline()

    # Convert Pandas DataFrame to NumPy Array
    training_data = training_data.to_numpy()

    # Step 2: Prepare Training & Testing Data
    print(" Preparing training and testing datasets...")

    training_data_len = int(config.TRAINING_DATA_FRAC * training_data.shape[0])

    # Extracting positive and negative class data
    pos_class_data = training_data[training_data[:, 0] == 1]
    neg_class_data = training_data[training_data[:, 0] == 0]

    # Ensure there's enough data for both classes
    min_class_size = min(len(pos_class_data), len(neg_class_data))
    train_size_per_class = min(training_data_len // 2, min_class_size)

    pos_class_train = pos_class_data[:train_size_per_class]
    neg_class_train = neg_class_data[:train_size_per_class]

    # Combine positive & negative class training data
    training_data = np.vstack((pos_class_train, neg_class_train))

    X_train = training_data[:, 1:]  # Features
    y_train = training_data[:, 0].reshape(-1, 1)  # Labels

    # Step 3: Train the Model
    print(" Training the model...")

    trained_params = training(config.EPSILON, X_train, y_train, config.TOL)

    # Step 4: Save the Model
    print(" Saving trained model...")

    os.makedirs(config.SAVED_MODEL_PATH, exist_ok=True)

    with open(os.path.join(config.SAVED_MODEL_PATH, config.SAVED_MODEL_FILE), "wb") as file_handle:
        pickle.dump(trained_params, file_handle)

    print("Model training complete! Saved at:", os.path.join(config.SAVED_MODEL_PATH, config.SAVED_MODEL_FILE))

if __name__ == "__main__":
    run_training()
