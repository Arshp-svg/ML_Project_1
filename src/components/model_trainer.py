import os
import sys
from dataclasses import dataclass
from catboost import CatBoostClassifier
from src.utils import evaluate_models

from sklearn.ensemble import (
   AdaBoostRegressor,
   GradientBoostingRegressor,
   RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train,y_train,X_test,y_test= (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models = {
                "LinearRegression": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "CatBoostClassifier": CatBoostClassifier(verbose=False)
            }
            
            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models)
            
            # best model score from the model report
            best_model_score=max(sorted(model_report.values()))
            
            
            # Get the best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model found", sys)
            
            logging.info(
                f"Best model found, model name: {best_model_name} and score: {best_model_score}"
            )
            
            save_object(
                obj=best_model,
                file_path=self.model_trainer_config.trained_model_file_path
            )
            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
                   
        except Exception as e:
            raise CustomException(e, sys)