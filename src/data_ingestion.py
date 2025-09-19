import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import logging
import yaml

# ensure the logs directory exists 
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration 
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)-> dict:
    '''Load Parameters from a YAML file'''
    try:
        with open(params_path,"r")as file:
            params = yaml.safe_load(file)
        logger.info("file loaded succesfully")
        return params
    except Exception as e:
        logger.error("error occured while loading the yaml file : %s",e)
        raise

def load_data(data_url:str)-> pd.DataFrame:
    # "Load data from csv file"
    try:
       df = pd.read_csv(data_url)
       logger.debug("Data loaded from %s",data_url)
       return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file %s",e)
        raise
    except Exception as e:
        logger.error("unexpected error occur while loading the file %s",e)
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    # preprocessing the data 
    try:
        return df
    except KeyError as e:
        logger.error("Missing column in data : %s",e)
        raise
    except Exception as e:
        logger.error("unexceptrd error during preprocessing: %s",e)
        raise 

def save_data(train:pd.DataFrame , test: pd.DataFrame, data_path:str)->None:
    # saving the train and test dataset:
    try:
        raw_data_path = os.path.join(data_path,"raw")
        os.makedirs(raw_data_path,exist_ok= True)
        train.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("train and test dataset save to %s",raw_data_path)
    except Exception as e :
        logger.error("unexpected error occur while saving the train and test dataset: %s",e)
        raise

def main():
    try :
        params = load_params("/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/params.yaml")
        test_size = params["data_ingestion"]['test_size']
        data_path = "/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/Customer-Churn-Records.csv"

        df = load_data(data_path)
        final_data = preprocess_data(df)
        train_data , test_data = train_test_split(final_data,test_size=test_size,random_state=2)
        save_data(train_data,test_data,data_path='./data')
    
    except Exception as e:
        logger.error("Failed to completer data Ingestion process: %s",e)
        raise


if __name__ == "__main__":
    main()