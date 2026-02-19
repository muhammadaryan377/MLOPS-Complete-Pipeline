import pandas as pd
import os
from sklearn.model_selection import train_test_split    
import logging
import yaml

log_dir = 'logs'

os.makedirs(log_dir, exist_ok=True)



logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load "Parmameters from yaml file"""

    try:
        with open(params_path, 'r') as yaml_file:
            params = yaml.safe_load(yaml_file)
            logger.debug(f"Parameters loaded successfully from {params_path}")
            return params
    except FileNotFoundError as e:
        logger.error(f"Parameters file not found:{e}")
        raise
    except yaml.YAMLError as e:
        logger.debug("Yaml Error while Loading Parameters")
        raise
    except Exception as e:
        logger.error(f"Unexpecter Error while loading parameters: {e}")
        raise


def load_data(data_path: str) -> pd.DataFrame:
    try:
        df  = pd.DataFrame(pd.read_csv(data_path))
        logger.debug(f"Data loaded successfully from {data_path}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data file: {e}")
        raise   
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data:{e}")
        raise
def PrePprocessdata(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1' : 'targer', 'v2' : 'text'}, inplace = True)
        logger.debug('Data preprocessing comleted')
        return df
    except KeyError as e:
        logger.erro(f"Missing columns in data frame: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data preprocessing: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index = False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index = False)
        logger.debug(f"Data saved succeessfully to {raw_data_path}")
    except Exception as e:
        print(f'Unexpected error while saving data: {e}')
        raise

def main():
        try:
           params = load_params('params.yaml')
           test_size = params['data_ingestion']['test_size']
         
           data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'
           df = load_data(data_path)
           df = PrePprocessdata(df)
           train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
           save_data(train_data, test_data, data_path= './data')
        except Exception as e:
            logger.error(f"Unexpected error in main function: {e}")


if __name__ == "__main__":
    main()

    


        