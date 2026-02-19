import os
import json
import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# ===============================
# Logging configuration
# ===============================
log_dirs = 'logs'
os.makedirs(log_dirs, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dirs, 'model_evaluation.log'))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ===============================
# Load model
# ===============================
def load_model(model_path: str):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            logger.debug(f"Model loaded successfully from {model_path}")
            return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

# ===============================
# Load CSV
# ===============================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path} | shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

# ===============================
# Load TF-IDF feature names
# ===============================
def load_feature_names(path: str):
    try:
        with open(path, 'r') as f:
            feature_names = json.load(f)
            logger.debug(f"TF-IDF feature names loaded from {path} | count: {len(feature_names)}")
            return feature_names
    except Exception as e:
        logger.error(f"Error loading feature names: {e}")
        raise

# ===============================
# Align test features
# ===============================
def align_features(df: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Ensure test data columns match training feature columns.
    Missing columns are filled with zeros.
    Extra columns are dropped.
    """
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]  # reorder
    return df

# ===============================
# Evaluate model
# ===============================
def evaluate_model(model, X_test: pd.DataFrame, y_test: np.array) -> dict:
    try:
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logger.debug(f"Evaluation metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise

# ===============================
# Save metrics
# ===============================
def save_metrics(metrics: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

# ===============================
# Main function
# ===============================
def main():
    try:
        # Load model
        model = load_model("models/random_forest_model.pkl")

        # Load test data
        test_data = load_data("data/features/test_tfidf.csv")
        test_data.fillna(0, inplace=True)

        # Load TF-IDF feature names from training
        feature_names = load_feature_names("models/train_feature_names.json")

        # Align test features
        X_test = align_features(test_data.drop(columns=['label'], errors='ignore'), feature_names)

        # Load target labels
        y_test = pd.read_csv("data/processed/test_processed.csv")['target'].values

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)

        # Save metrics
        save_metrics(metrics, "metrics/model_metrics.json")

    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise

if __name__ == "__main__":
    main()
