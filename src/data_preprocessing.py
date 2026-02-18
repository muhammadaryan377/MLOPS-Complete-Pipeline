import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')


log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def transform_text(text: str) -> str:
    try:
        """
        Trnasform the input text by removing punctuation, removing stop words, converting to lowercase , tokenizing and stemming
        """
        ps = PorterStemmer()
        
        # Convert text to lowercase
        text = text.lower()
        # Tokenize the text
        text = nltk.word_tokenize(text)

        #Remove non non-alphanumeric token
        text = [word for word in text if word.isalnum()]

        #Remove the stop words and punctation
        text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
       
        #stemming the text
        text = [ps.stem(word) for word in text]
        # joins the tokens back to a single string
        text = ' '.join(text)
        return text
    except Exception as e:
        logger.error(f"Error while transforming the text: {e}")
        raise

def preprocess_data(df: pd.DataFrame, text_columns: str = 'text', target_column: str = 'target') -> pd.DataFrame:
    """ 
       Preprocessing the DataFame by enciding the target variable and removing the duplicates and transforming the text columns
    """
    try:
        logger.debug("Starting data preprocessing")
        # Encode the target column
        label_encoder = LabelEncoder()
        df[target_column] = label_encoder.fit_transform(df[target_column])
        logger.debug("Target column encoded successfully")

        # remove duplicates
        df.drop_duplicates(keep = 'first', inplace = True)
        logger.debug("Duplicates removed successfully")

        #Aplly the text transformation to the text columns
        df[text_columns] = df[text_columns].apply(transform_text)
        logger.debug("Text tranformation appled sunccessfully")
        return df
    except KeyError as e:
        logger.error(f"Key error during data preprocessinf: {e}")
        raise

    except Exception as e:
        logger.error(f"Unexpected error during data preprocessing: {e}")
        raise


def main(text_columns: str = 'text', target_columns: str = 'target') -> None:
    """
    DocstriMng for main
    
    :param text_columns: Description
    :type text_columns: str
    :param target_columns: Description
    :type target_columns: str
    Main function to load the raw data, prepocess the data and save the preprocessed data
    """     

    try:
        # Fetech the data from data/raw
        train_data = pd.DataFrame(pd.read_csv(os.path.join('data', 'raw', 'train.csv')))
        test_data = pd.DataFrame(pd.read_csv(os.path.join('data', 'raw', 'test.csv')))
        logger.debug("Data loaded succesfully for preprocessing")

        # preprocess the data
        train_processed_data = preprocess_data(train_data, text_columns=text_columns, target_column=target_columns)
        test_processed_data = preprocess_data(test_data, text_columns=text_columns, target_column=target_columns)
  
        logger.debug("Data preprocessed successfully")
        
        data_path = os.path.join('data', 'processed')
        os.makedirs(data_path, exist_ok=True)
        

        #save the preprocessed data to data/processed

        train_processed_data.to_csv(os.path.join('data', 'processed', 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join('data', 'processed', 'test_processed.csv'), index=False)
        logger.debug("Preprocessed data saved successfully")
        
    except FileNotFoundError as e:
         logger.error(f"Data file not found during preprocessing: {e}")
         raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data file during preprocessing: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing data file during preprocessing: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during data preprocessing: {e}")
        raise
        

