import os 
import logging
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder,StandardScaler

# ensure that the log dir exists
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# logging configuration 
logger = logging.getLogger("Datap_preprocessing")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path )
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def normalize(df,columns):
    try:
        scaler = StandardScaler()
        df[columns]=scaler.fit_transform(df[columns])
        logger.debug("columns have been normalized")
        return df

    except Exception as e:
        logger.error("unexpected Error occured: %s",e)
        raise


def oneHot(df,columns):
    """
    funtion for converting text to numbers 
    """

    try:
        ohe= OneHotEncoder(drop= "first")
        encoded = ohe.fit_transform(df[columns]).toarray()

        encoded_df = pd.DataFrame(
            encoded,
            columns= ohe.get_feature_names_out(columns),
            index=df.index
        )

        df=df.drop(columns,axis=1).join(encoded_df)
        logger.info("dataFrame created with encoded values")
        return df

    except Exception as e:
        logger.error("Unexpected error oxxured during OneHotEncoding : %s",e)
        raise

def remove_columns(df,columns):
    """
    Removes useless columns or columns we want to remove
    """
    try:
        df=df.drop(columns=columns,axis = 1)
        return df
        logger.info(f'{columns} are removed from the dataset')

    except Exception as e:
        logger.error("Error occured while removing the columns: %s",e)
        raise


def main():
    try: 
        train_data = pd.read_csv('/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/data/raw/train.csv')
        test_data = pd.read_csv('/Users/sarthaktyagi/Desktop/30days-3oprojects/youtubeMLOps/data/raw/test.csv')
        logger.info("dataset loaded succesfully")

        # drop the columns 
        drop_col = ['RowNumber','CustomerId','Surname']
        train_dropped = remove_columns(train_data,columns=drop_col)
        test_dropped = remove_columns(test_data,columns=drop_col)

        # data transformation 
        numerical_col = []
        categorical_col = []

        for col,dtype in train_dropped.dtypes.items():
            if pd.api.types.is_numeric_dtype(dtype):
                numerical_col.append(col)
            elif pd.api.types.is_object_dtype(dtype):
                categorical_col.append(col)

        numerical_col.remove("Exited")
        train_preprocess = normalize(train_dropped,numerical_col)
        test_preprocess = normalize(test_dropped,numerical_col)

        train_final = oneHot(train_preprocess,categorical_col)
        test_final = oneHot(test_preprocess,categorical_col)
        logger.info("train and test data, normalized and encoded")
        
        # saving the data 
        dir_path = os.path.join("./data","interdim")
        os.makedirs(dir_path,exist_ok=True)
        logger.info("%s created",dir_path)

        train_final.to_csv(os.path.join(dir_path,"train_final.csv"),index = False)
        test_final.to_csv(os.path.join(dir_path,"test_final.csv"),index = False)
        logger.debug("Both train and test data saved ")

    except FileExistsError as e:
        logger.error("file not found : %s",e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("data is empty : %s",e)
        raise
    except Exception as e:
        logger.error("Unexcepted error Occured : %s",e)
        raise

if __name__ == "__main__":
    main()
