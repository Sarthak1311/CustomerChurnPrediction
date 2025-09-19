import pickle 
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
import json 
import os
import logging

log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

# logging configuration 
logger = logging.getLogger("Model Evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir,"model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(url:str)->pd.DataFrame:
    """
    load the csv aand return a dataframe"""
    try:
        df = pd.read_csv(url)
        logger.info("data loaded succesfully")
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file :%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occured while loading the data : %s",e)
        raise

def load_model(file_path:str):
    """Loads the training model for the file"""
    try:
        with open(file_path,"rb") as file:
            model = pickle.load(file)
        logger.info("loaded the model succesfully from %s",file_path)
        return model
    except FileNotFoundError as e:
        logger.error("file Not found : %s",file_path)
        raise
    except Exception as e :
        logger.error("unexpected error occured while loading the model: %s",e)
        raise


def evaluation_model(clf,X_test,y_test)->dict:
    '''evaluate the model and return a evaluation metrics'''
    try:
        y_pred= clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall= recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,y_pred_proba[:, 1])

        metrics ={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'f1_score':f1,
            'roc_auc':roc_auc
        }

        logger.info("model evaluation metrics calculated and created succesfully ")
        return metrics
    except Exception as e:
        logger.error("unexpected error occured while calculating evaluation metrics for the model: %s",e)
        raise

def save_metrics(metrics:dict,file_path:str)->None :
    """save the evaluation metrics to a json file"""
    try:
        # ensure the directory is there 
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path,"w") as file:
            json.dump(metrics,file,indent=4)
        logger.debug("metrics saved to %s",file_path)
    except Exception as e:
        logger.error("unexpected error occured while saving the metrics to json : %s",e)
        raise

def main():
    try:
        # load the data and the model 
        test_data = pd.read_csv("/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/data/interdim/test_final.csv")
        model = load_model("/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/model/model.pkl")

        X_test = test_data.drop(['Exited'],axis =1)
        y_test = test_data['Exited'].astype(int)

        # calculated metrics 
        metrics = evaluation_model(model,X_test,y_test)

        save_metrics(metrics,"reports/metrics.json")

    except FileNotFoundError as e:
        logger.error("file Not Found : %s",e)
        raise
    except Exception as e:
        logger.error("Failed to complete the model evaluation process :%s",e)
        raise
        
if __name__ == "__main__":
    main()