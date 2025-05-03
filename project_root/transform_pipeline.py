import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import config

def complete_pipeline():
    # Load the dataset
    data_path = os.path.join(config.ROOT_DIR_PATH, config.DATA_DIR, config.FILENAME)
    data = pd.read_csv(data_path)

    # Drop duplicates
    data.drop_duplicates(inplace=True)

    # Handle missing values (drop rows with NaN values)
    data.dropna(inplace=True)

    # Select refined features
    data = data[config.REFINED_COLUMNS]

    # Encode target variable (M → 1, B → 0) using map()
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})


    # Save cleaned dataset (optional for debugging or record-keeping)
    cleaned_path = os.path.join(config.DATA_DIR, "cleaned_full_data.csv")
    data.to_csv(cleaned_path, index=False)

    # Split into train & test sets using scikit-learn
    train_data, test_data = train_test_split(
        data,
        test_size=1 - config.TRAINING_DATA_FRAC,
        random_state=42,
        stratify=data["diagnosis"]  # preserves class distribution
    )

    # Separate features & target
    X_train = train_data.drop(columns=["diagnosis"]).values
    y_train = train_data["diagnosis"].values.reshape(-1, 1)

    X_test = test_data.drop(columns=["diagnosis"]).values
    y_test = test_data["diagnosis"].values.reshape(-1, 1)

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

    print("Data preprocessing complete!")

    return train_df, test_df

if __name__ == "__main__":
    complete_pipeline()
