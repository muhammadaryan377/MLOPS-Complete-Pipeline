import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# ===============================
# Logging configuration
# ===============================
log_dirs = 'logs'
os.makedirs(log_dirs, exist_ok=True)

logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dirs, 'feature_engineering.log'))

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ===============================
# Load YAML parameters
# ===============================
def load_params(params_path: str) -> dict:
    """
    Loads parameters from a YAML file.
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            logger.debug(f"Parameters loaded successfully from {params_path}")
            return params
    except Exception as e:
        logger.error(f"Error loading YAML parameters: {e}")
        raise

# ===============================
# Load CSV data
# ===============================
def load_data(data_path: str) -> pd.DataFrame:
    """
    Loads CSV data and fills missing text with empty strings.
    """
    try:
        df = pd.read_csv(data_path)
        df.fillna('', inplace=True)
        logger.debug(f"Data loaded successfully from {data_path} | shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise

# ===============================
# Apply TF-IDF vectorization
# ===============================
def apply_tfidf_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int):
    """
    Apply TF-IDF vectorization to the 'text' column.
    Returns train/test DataFrames with TF-IDF features + labels
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        train_tfidf = vectorizer.fit_transform(train_data['text'])
        test_tfidf = vectorizer.transform(test_data['text'])

        # Convert to DataFrame with feature names
        feature_names = vectorizer.get_feature_names_out()
        train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=feature_names)
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray(), columns=feature_names)

        # Add label column
        train_tfidf_df['label'] = train_data['target'].values
        test_tfidf_df['label'] = test_data['target'].values

        logger.debug(f"TF-IDF applied | Train shape: {train_tfidf_df.shape} | Test shape: {test_tfidf_df.shape}")
        return train_tfidf_df, test_tfidf_df, feature_names

    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {e}")
        raise

# ===============================
# Save DataFrame to CSV
# ===============================
def save_data(df: pd.DataFrame, data_path: str) -> None:
    """
    Saves a DataFrame to CSV, creating directories if missing.
    """
    try:
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        logger.debug(f"Data saved to {data_path} | shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error saving data to {data_path}: {e}")
        raise

# ===============================
# Save feature names
# ===============================
def save_feature_names(feature_names, path):
    """
    Save TF-IDF feature names for model evaluation.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(feature_names.tolist(), f)
        logger.debug(f"TF-IDF feature names saved to {path}")
    except Exception as e:
        logger.error(f"Error saving feature names: {e}")
        raise

# ===============================
# Main function
# ===============================
def main():
    try:
        # Load parameters or use default
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
       

        # Load processed train/test CSVs
        train_data = load_data(os.path.join('data', 'processed', 'train_processed.csv'))
        test_data = load_data(os.path.join('data', 'processed', 'test_processed.csv'))

        # Apply TF-IDF
        train_tfidf_df, test_tfidf_df, feature_names = apply_tfidf_vectorization(train_data, test_data, max_features)

        # Save features
        save_data(train_tfidf_df, os.path.join('data', 'features', 'train_tfidf.csv'))
        save_data(test_tfidf_df, os.path.join('data', 'features', 'test_tfidf.csv'))

        # Save TF-IDF feature names for evaluation
        save_feature_names(feature_names, os.path.join('models', 'train_feature_names.json'))

    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
