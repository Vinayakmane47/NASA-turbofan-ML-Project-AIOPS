from Turbofan.exception import TurboException 
from Turbofan.logger import logging 
from Turbofan.entity.config_entity import ModelTrainerConfig 
from Turbofan.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact , DataTransformationArtifact , ModelTrainerArtifact
import os,sys 
import pandas as pd 
import numpy as np 
from Turbofan.util.util import * 
from typing import List 
from Turbofan.entity.model_factory import MetricInfoArtifact, ModelFactory,GridSearchedBestModel, evaluate_regression_model



class TurboPredictor: 
    
    def __init__(self,pred_obj ,preproccing_obj): 
        self.pred_obj = pred_obj 
        self.preproccing_obj = preproccing_obj 
        
    def predict(self,x):
        transform_features = self.preproccing_obj.transform(x)
        return self.pred_obj.predict(transform_features)
    
    def __repr__(self): 
        return f"{type(self.pred_obj).__name__}()"
    
    def __str__(self): 
        return f"{type(self.pred_obj).__name__}()"



class ModelTrainer: 


    def __init__(self, model_trainer_config:ModelTrainerConfig, 
                        data_transformation_artifact:DataTransformationArtifact) -> None:
        try : 
            logging.info(f"{'=='*20} Model Trainer Started {'=='*20}") 
            self.model_trainer_config = model_trainer_config 
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e : 
            raise TurboException(e,sys)

    def initiate_model_trainer(self)->ModelTrainerArtifact: 

        try: 
            logging.info("Loading train and test dataset which are transformed previously")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path 
            transformed_test_file_path  = self.data_transformation_artifact.transformed_test_file_path 

            ## Loading numpy arrays : 
            train_arr = load_numpy_array(file_path=transformed_train_file_path)
            test_arr = load_numpy_array(file_path=transformed_test_file_path)

            logging.info("Splitting train and test arrays ")
            x_train_arr = train_arr[:,:-1]
            y_train_arr = train_arr[:,-1]
            x_test_arr = test_arr[:,:-1]
            y_test_arr = test_arr[:,-1]

            logging.info("Extracting Model config File path ")
            model_config_file_path = self.model_trainer_config.model_config_file_path 

            logging.info(f"getting model factory object using model config path {model_config_file_path}")
            model_factory = ModelFactory(model_config_path=model_config_file_path)
            
            ## Base Accuracy : 
            base_accuracy = self.model_trainer_config.base_accuracy 
            logging.info(f"Our Expected Accuracy is {base_accuracy}")

            logging.info("Getting Best Model on by doing grid search on train dataset ")
            ## best model on train dataset only by comparing all grid search models in grid search model list 
            best_model = model_factory.get_best_model(X=x_train_arr,y=y_train_arr)

            logging.info(f"Best Model Found on Training Dataset {best_model}")

            logging.info(f"Getting list of trained models on train dataset previously by grid search")
            ## This model list have best params which are obtained by grid searching 
            get_searched_model_list : List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in get_searched_model_list]

            logging.info(f"Evaluating all grid searched models on train and test dataset") 
            metric_info :MetricInfoArtifact = evaluate_regression_model(model_list=model_list , 
                                                    X_train=x_train_arr , y_train=y_train_arr , 
                                                    X_test=x_test_arr ,y_test=y_test_arr ,base_accuracy=base_accuracy)

            preproccesing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object 
            trained_model_file_path = self.model_trainer_config.trained_model_file_path 

            Turbofan_model = TurboPredictor(pred_obj=model_object,preproccing_obj=preproccesing_obj)

            logging.info(f"Saving Turbofan Model at path : {trained_model_file_path}")

            save_object(file_path=trained_model_file_path,obj=Turbofan_model)

            model_trainer_artifact = ModelTrainerArtifact(is_trained=True , 
                        message="Model Trained Succesfully ", 
                        trained_model_file_path=trained_model_file_path , 
                        train_rmse=metric_info.train_rmse , 
                        test_rmse=metric_info.test_rmse , 
                        train_accuracy=metric_info.train_accuracy , 
                        test_accuracy= metric_info.test_accuracy , 
                        model_accuracy= metric_info.model_accuracy )

            logging.info(f"Model Trainer Artifact {model_trainer_artifact}")

            return model_trainer_artifact
        except Exception as e : 
            raise TurboException(e,sys)
