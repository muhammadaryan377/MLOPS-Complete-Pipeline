import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Set up logging
log_dirs = 'logs'
os.makedirs(log_dirs, exist_ok= True)

#logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler(os.path.join(log_dirs, 'feature_engineering.log'))

console_handler.setLevel(logging.DEBUG)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Function to load parameters from yaml file

def load_params(params_path: str) -> dict:
    """"
    
    Loads parameters from a YAML file.
    Args:
        params_path (str): Path to the YAML parameters file.
    Returns:
        dict: The loaded parameters.
    """

    try:
        with open(params_path, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
            logger.debug(f"Paramters loaded successfully from {params_path}")
            return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML Error while loading parameters: {e}")
        raise   
    except Exception as e:
        logger.error(f"Unexpected error while loading parameters: {e}")
        raise


# load the data
def load_data(data_path: str) -> pd.DataFrame:
    """"
    Loads data from a CSV file.
    Args:
        data_path (str): Path to the CSV data file."""
    try:
        df = pd.DataFrame(pd.read_csv(data_path))
        df.fillna('', inplace=True)
        logger.debug(f"Data loaded successfully from {data_path}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data file: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parasing data file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

## apply tfidf vectorization to the text data
def apply_tfidf_vectorization(train_data: pd.DataFrame, test_data: pd.DataFrame, max_feature: int) -> tuple:
    """
    Applies TF-IDF vectorization to the specified text column in the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
       
        max_features (int): The maximum number of features to be extracted by the TF-IDF vectorizer."""
    
    try:
        tfidf_vectorize = TfidfVectorizer(max_features= max_feature)
        train_tfidf = tfidf_vectorize.fit_transform(train_data['text'])
        test_tfidf = tfidf_vectorize.transform(test_data['text'])

        train_tfidf_df = pd.DataFrame(train_tfidf.toarray())
        test_tfidf_df = pd.DataFrame(test_tfidf.toarray())

        train_tfidf_df['label'] = train_data['target'].values
        test_tfidf_df['label'] = test_data['target'].values

        logger.debug("TF-IDF vectorization applied successfully")
        return train_tfidf_df, test_tfidf_df
        
    except KeyError as e:
        logger.error(f"Key error during TF-IDF vectorization: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during TF-IDF vectorization: {e}")
        raise

def save_data(df: pd.DataFrame, data_path: str) -> None:
    """
    Saves the DataFrame to a CSV file.
    """    
    try:

       
        df.to_csv(data_path, index=False)
        logger.debug(f"Data saved successfully to {data_path}") 
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}")
        raise
        
def main():
    try:
        #params = load_params(os.path.join('params.yaml'))
        #max_features = params['feature_engineering']['max_features']
        max_features = 5000
        train_data = load_data(os.path.join('data', 'processed', 'train_processed.csv'))
        test_data = load_data(os.path.join('data', 'processed', 'test_processed.csv'))  
        train_tfidf_df, test_tfidf_df = apply_tfidf_vectorization(train_data, test_data, max_features)
        save_data(train_tfidf_df, os.path.join('data', 'features', 'train_tfidf.csv')) 
        save_data(test_tfidf_df, os.path.join('data', 'features', 'test_tfidf.csv'))
    except Exception as e:
        logger.error(f"Unexpected error in main function: {e}")
        raise
if __name__ == "__main__":
    main() 