# for data transformation

import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    # path to store model.
    preprocessor_ob_file_path= os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        # Initialize an instance of DataTransformationConfig class
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        get_data_transformer_object function is responsible for data transformer for different types of data.
        '''
        try:
            # Define numerical and categorical columns
            numerical_columns= ['reading score', 'writing score']
            categorical_columns= ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            # Define numerical pipeline to handle numerical data
            num_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Impute missing values with median
                    ("scalar", StandardScaler(with_mean=False))  # Scale numerical features
                ]
            )

            # Define categorical pipeline to handle categorical data
            cat_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Impute missing values with most frequent value
                    ("one_hot_encoder", OneHotEncoder()),  # Perform one-hot encoding
                    ("scaler", StandardScaler(with_mean=False))  # Scale categorical features
                ]
            )

            # Log messages for pipeline completion
            logging.info("Numerical Columns Standard Scaling completed!")
            logging.info("Categorical Columns encoding completed!")

            # Combine both pipelines using ColumnTransformer
            preprocessor= ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply numerical pipeline to numerical columns
                    ("cat_pipeline", cat_pipeline, categorical_columns),  # Apply categorical pipeline to categorical columns
                ]
            )

            return preprocessor

        except Exception as e:
            # Raise CustomException if an error occurs
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read training and testing data from CSV files
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            # Log message for successful data reading
            logging.info("Read Train and Test Data Completed!")

            # Obtain preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj= self.get_data_transformer_object()

            # Define target column name and numerical columns
            target_column_name= "math score"
            numerical_columns= ['reading score', 'writing score']

            # Separate input and target features for training data
            input_feature_train_df= train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df= train_df[target_column_name]

            # Separate input and target features for testing data
            input_feature_test_df= test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df= test_df[target_column_name]

            # Log message for preprocessing application
            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # Apply preprocessing object to training and testing dataframes
            input_feature_train_arr= preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessing_obj.transform(input_feature_test_df)

            # Concatenate input features and target features for training data
            train_arr= np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            # Concatenate input features and target features for testing data
            test_arr= np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Log message for preprocessing object saving
            logging.info("Saved Preprocessing Object.")

            # Saving pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            # Return processed training and testing data along with preprocessing object file path
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_ob_file_path)

        except Exception as e:
            raise CustomException(e, sys)
        

