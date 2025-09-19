import pickle
import numpy as np
import pandas as pd 
import logging 
import os 
from sklearn.ensemble import RandomForestClassifier
import yaml
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging confriguation 
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir,"model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

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

def load_data(file_path:str)->pd.DataFrame:

    '''
    load the csv file 
    return dataframe
    '''

    try:
        df = pd.read_csv(file_path)
        logger.info("data have been succesfully loaded")
        return df
    except FileExistsError as e:
        logger.error("file does not exists : %s",e)
        raise
    except pd.errors.ParserError as e :
        logger.error("Failed to parse the csv file : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected errror occured : %s",e)
        raise

def model_train(X_train,y_train,params:dict) -> RandomForestClassifier:
    '''
    Train the model
    :param X_train -> training features 
    :param y_train -> target features 
    :params params -> Dictionary of hyperparameters 
    return a trained Random forest Regressior

    '''
    try:    
        if X_train.shape[0]!= y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be same")

        logger.info("Initializing Random forest model with parameter : %s",params)
        clf= RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])

        logger.info("Model training started with %d samples ",X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.info("Model training completed")
        
        return clf 

    except ValueError as e:
        logger.error("Value error during model_training : %s",e)
        raise

    except Exception as e:
        logger.error("Error during model Training : %s",e)
        raise

def model_save(model, file_path: str) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)   # only create the parent directory
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.info("Model saved to the file location: %s", file_path)
    except Exception as e:
        logger.error("Error occurred during saving the model: %s", e)
        raise

def main():
    try:
        params = load_params("/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/params.yaml")

        df = load_data('/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/data/interdim/train_final.csv')
        params= {
            "n_estimators":params['model_training']['n_estimators'],
            "random_state":params['model_training']['random_state']
        }
        X_train = df.drop(['Exited'],axis =1)
        y_train= df['Exited'].astype(int)

        clf = model_train(X_train=X_train,y_train=y_train,params=params)

        model_save_path = 'model/model.pkl'

        model_save(model=clf,file_path=model_save_path)
    except Exception as e :
        logger.error("Failed to complete the model training process: %s",e)
        raise

if __name__ == "__main__":
    main()