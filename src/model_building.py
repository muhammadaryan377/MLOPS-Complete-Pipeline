import os
import logging 
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


# Set up logging
log_dir = 'logs'

os.makedirs(log_dir, exist_ok= True)
logger = logging.getLogger('model_buliding')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
path = os.path.join(log_dir, 'model_buliding.log')

file_handler = logging.FileHandler(path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s -%(message)s")

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """
    Docstring for load_params
    
    :param params_path: Description
    :type params_path: str
    :return: Description
    :rtype: dict
    """

    try:
        with open(params_path, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
            logger.debug(f"Parameters loaded successfully from {params_path}")
            return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML Error while loading parametrs: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading parameters: {e}")
        raise


def load_data(file: str) -> pd.DataFrame:
    """"
    
    :param file: Path to the data file
    :type file: str
    :return: DataFrame loaded from the file
    :rtype: pd.DataFrame
    """    
    try:
        df = pd.read_csv(file)
        logger.debug(f"Data loaded successfully from {file}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data in file {file}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file {file}: {e}")
        raise
    except Exception as e:  
        logger.error(f"Unexpected error while loading data from {file}: {e}")
        raise


def train_model(X_tain: np.array, y_train: np.array, model_params: dict) -> RandomForestClassifier:
    """
    Docstring for train_model
    
    :param X_tain: Description
    :type X_tain: np.array
    :param y_array: Description
    :type y_array: np.array
    :param model_params: Description
    :type model_params: dict
    :return: Description
    :rtype: RandomForestClassifier
    """

    try:
       if X_tain.shape[0] != y_train.shape[0]:
           raise ValueError("Number of samples in X_train and y_train must be the same")
       logger.debug("Starting model training")
       model = RandomForestClassifier(n_estimators= 5, max_depth = 5, random_state= 42)
       model.fit(X_tain, y_train)
       logger.debug("Model training completed sucessfully")
       return model
    except ValueError as e:
        logger.error(f"Value error during model training: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model training: {e}")
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """
    Docstring for save_model
    
    :param model: Description
    :type model: RandomForestClassifier
    :param model_path: Description
    :type model_path: str
    """
    try:

        # Ensure the directory for the model exists
        os.makedirs(os.path.dirname(model_path), exist_ok = True)

        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)

        logger.debug(f"Model saved successfully to {model_path}")
    except FileNotFoundError as e:
         logger.error(f"Model path not found: {e}")
         raise
    except Exception as e:
        logger.error(f"Unexpected error while saving model: {e}")
        raise   


def main():
    try:
        #params = load_params(os.path.join('params.yaml'))
        train_data = load_data(os.path.join('data', 'features', 'train_tfidf.csv'))
        X_train = train_data.drop(columns=['label']).values
        y_train = train_data['label'].values

        clf = train_model(X_train, y_train, model_params = None)
        save_model(clf, os.path.join('models', 'random_forest_model.pkl'))
    except Exception as e:  
        logger.error(f"Unexpected error in main function: {e}")
        raise       

if __name__ == "__main__":
    main()
