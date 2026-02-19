import os
import logging
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# ===============================
# Logging configuration
# ===============================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dir, 'model_building.log'))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ===============================
# Load parameters from YAML (optional)
# ===============================
def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
            logger.debug(f"Parameters loaded successfully from {params_path}")
            return params
    except Exception as e:
        logger.error(f"Error loading YAML parameters: {e}")
        raise

# ===============================
# Load CSV data
# ===============================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path} | shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

# ===============================
# Train Random Forest model
# ===============================
def train_model(X_train: np.array, y_train: np.array, model_params: dict = None) -> RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match")

        logger.debug(f"Training RandomForestClassifier | Samples: {X_train.shape[0]} | Features: {X_train.shape[1]}")
        
        # Use parameters if provided
        if model_params is None:
            model_params = {'n_estimators': 5, 'max_depth': 5, 'random_state': 42}

        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        logger.debug("Model training completed successfully")
        return model
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

# ===============================
# Save model to file
# ===============================
def save_model(model: RandomForestClassifier, model_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")
        raise

# ===============================
# Main function
# ===============================
def main():
    try:
        # Load processed TF-IDF features
        train_data = load_data(os.path.join('data', 'features', 'train_tfidf.csv'))
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values
        logger.debug(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")

        # Train model
        clf = train_model(X_train, y_train)

        # Save trained model
        save_model(clf, os.path.join('models', 'random_forest_model.pkl'))

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise

if __name__ == "__main__":
    main()
