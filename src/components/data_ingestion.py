# This file will contain code for/related to reading data

import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig


# used to create class variables
@dataclass                # directly declare class variables w/o __init__ method. Use this if you only want to define vars in a class
class DataIngestionConfig:
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()            

    def initiate_data_ingestion(self):             # to read the data - from files, db, urls etc.
        logging.info("Entered the data ingestion method/component")
        try:
            df= pd.read_csv("notebook/StudentsPerformance.csv")
            logging.info("Reading the Dataset as Dataframe.")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Split Initiated.")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        


if __name__ == "__main__":
    obj= DataIngestion()
    train_data, test_data, raw_data= obj.initiate_data_ingestion()

    data_transformation= DataTransformation()
    data_transformation.initiate_data_transformation(train_path=train_data, test_path=test_data)

        
