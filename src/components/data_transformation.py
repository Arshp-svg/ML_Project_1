import sys
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function is responsible for data transformation.
        It handles both numerical and categorical data preprocessing."""
        
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            #Pipeline to handle both numerical and categorical data
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            #Pipeline for categorical data
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scalar', StandardScaler(with_mean=False))
            ])
            
            logging.info("Numerical columns: %s", numerical_columns)
            logging.info("Categorical columns: %s", categorical_columns)
            
            #ColumnTransformer to apply the pipelines to the respective columns
            preprocessor= ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )
            
            return preprocessor
            
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
    # This function initiates the data transformation process.
    # It reads the train and test data, applies the preprocessing object
        
    def initiate_data_transformation(self, train_path, test_path):
        
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Train and test data loaded successfully.")
            
            logging.info("obtaining preprocessing object.")
            
            preprocessor_obj = self.get_data_transformer_object()
            
            targe_column_name = 'math_score'
            numerical_columns = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=[targe_column_name], axis=1)
            target_feature_train_df = train_df[targe_column_name]
            
            input_feature_test_df = test_df.drop(columns=[targe_column_name], axis=1)
            target_feature_test_df = test_df[targe_column_name]
            
            logging.info("Applying preprocessing object on training and testing dataframes.")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            
            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
            
            
        except Exception as e:
            raise CustomException(e, sys)
        