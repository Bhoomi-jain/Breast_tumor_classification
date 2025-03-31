import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from config import config

def complete_pipeline():
    print("🔹 Loading dataset...")

    # Load the dataset
    data_path = os.path.join(config.ROOT_DIR_PATH, config.DATA_DIR, config.FILENAME)
    data = pd.read_csv(data_path)

    print("🔹 Preprocessing data...")

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    # Handle missing values (drop rows with NaN values)
    data.dropna(inplace=True)

    # Select refined features
    data = data[config.REFINED_COLUMNS]

    # Encode target variable (M → 1, B → 0)
    data["diagnosis"].replace({"M": 1, "B": 0}, inplace=True)

    # Split into train & test sets
    training_data_len = int(config.TRAINING_DATA_FRAC * len(data))
    train_data = data.iloc[:training_data_len]
    test_data = data.iloc[training_data_len:]

    # Separate features & target
    X_train = train_data.drop(columns=["diagnosis"]).values
    y_train = train_data["diagnosis"].values.reshape(-1, 1)

    X_test = test_data.drop(columns=["diagnosis"]).values
    y_test = test_data["diagnosis"].values.reshape(-1, 1)

    print("🔹 Normalizing features...")

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save scaler for future use
    os.makedirs(config.SAVED_MODEL_PATH, exist_ok=True)
    scaler_path = os.path.join(config.SAVED_MODEL_PATH, "scaler.pkl")
    
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)

    # Save processed datasets
    train_df = pd.DataFrame(np.hstack((y_train, X_train)), columns=["diagnosis"] + config.REFINED_COLUMNS[1:])
    test_df = pd.DataFrame(np.hstack((y_test, X_test)), columns=["diagnosis"] + config.REFINED_COLUMNS[1:])

    train_df.to_csv(os.path.join(config.DATA_DIR, config.TRAINING_DATA_FILENAME), index=False)
    test_df.to_csv(os.path.join(config.DATA_DIR, config.TESTING_DATA_FILENAME), index=False)

    print("✅ Data preprocessing complete!")
    
    return train_df, test_df

if __name__ == "__main__":
    complete_pipeline()
