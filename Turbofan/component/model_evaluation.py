from Turbofan.component.model_trainer import TurboPredictor
from Turbofan.exception import TurboException 
from Turbofan.logger import logging 
from Turbofan.entity.config_entity import ModelEvaluationConfig 
from Turbofan.entity.artifact_entity import DataIngestionArtifact ,DataValidationArtifact,DataTransformationArtifact , \
    ModelEvaluationArtifact ,ModelTrainerArtifact
from Turbofan.constant import * 
from Turbofan.entity.model_factory import evaluate_regression_model 
from Turbofan.util.util import * 
import os,sys 
import pandas as pd 
import numpy as np 


class ModelEvaluation : 


    def __init__(self , model_evaluation_config:ModelEvaluationConfig , 
                        data_ingestion_artifact:DataIngestionArtifact , 
                        data_validation_artifact:DataValidationArtifact, 
                        model_trainer_artifact:ModelTrainerArtifact
                ) -> None:
        try : 
            logging.info( "Data Evaluation is started ")
            self.model_evaluation_config = model_evaluation_config 
            self.data_ingestion_artifact = data_ingestion_artifact 
            self.data_validation_artifact = data_validation_artifact 
            self.model_trainer_artifact = model_trainer_artifact



        except Exception as e : 
            raise TurboException(e,sys)




    def get_best_model(self): 
        try : 
            model = None
            model_evaluation_file_path = self.model_evaluation_config.model_evaluation_file_path

            if not os.path.exists(model_evaluation_file_path):
                create_yaml_file(file_path=model_evaluation_file_path,
                                )
                return model
            model_eval_file_content = read_yaml_file(file_path=model_evaluation_file_path)

            model_eval_file_content = dict() if model_eval_file_content is None else model_eval_file_content

            if BEST_MODEL_KEY not in model_eval_file_content:
                return model

            model = load_object(file_path=model_eval_file_content[BEST_MODEL_KEY][MODEL_PATH_KEY])
            return model

        except Exception as e : 
            raise TurboException(e,sys)


    def update_evaluation_report(self,model_evaluation_artifact:ModelEvaluationArtifact): 
        try : 
            eval_file_path = self.model_evaluation_config.model_evaluation_file_path
            model_eval_content = read_yaml_file(file_path=eval_file_path)
            model_eval_content = dict() if model_eval_content is None else model_eval_content
            
            
            previous_best_model = None
            if BEST_MODEL_KEY in model_eval_content:
                previous_best_model = model_eval_content[BEST_MODEL_KEY]

            logging.info(f"Previous eval result: {model_eval_content}")
            eval_result = {
                BEST_MODEL_KEY: {
                    MODEL_PATH_KEY: model_evaluation_artifact.evaluated_model_path,
                }
            }

            if previous_best_model is not None:
                model_history = {self.model_evaluation_config.time_stamp: previous_best_model}
                if HISTORY_KEY not in model_eval_content:
                    history = {HISTORY_KEY: model_history}
                    eval_result.update(history)
                else:
                    model_eval_content[HISTORY_KEY].update(model_history)

            model_eval_content.update(eval_result)
            logging.info(f"Updated eval result:{model_eval_content}")
            create_yaml_file(file_path=eval_file_path, data=model_eval_content)
        except Exception as e : 
            raise TurboException(e,sys)

    
    def initiate_model_evaluation(self)->ModelEvaluationArtifact: 
        try : 
            logging.info("Initiating ModelEvaluation ")
            ## loading train and test dataset from ingestion artifact 
            train_data__file_path = self.data_ingestion_artifact.train_file_path 
            test_data_file_path = self.data_ingestion_artifact.test_file_path 
            schema_file_path = self.data_validation_artifact.schema_file_path 
            train_df = load_data(file_path=train_data__file_path,schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_data_file_path,schema_file_path=schema_file_path)

            ## loading trained model 
            trained_model_file_path = self.model_trainer_artifact.trained_model_file_path 
            train_model_object = load_object(file_path=trained_model_file_path)

            schema_info = read_yaml_file(file_path=schema_file_path)
            target_column_name = schema_info[TARGET_COLUMN_KEY]

            ## Preparing x and y features from train and test dataset 
            x_train_df = train_df.drop(columns=target_column_name)
            x_test_df = test_df.drop(columns=target_column_name)
            y_train_arr = np.array(train_df[target_column_name])
            y_test_arr = np.array(test_df[target_column_name])

            ## getting best model from model evaluation file 
            model:TurboPredictor = self.get_best_model()
            # for first time the model is None : 
            if model is None : 
                logging.info(f"Not found any existing model hence we will accept trained model ")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path , 
                                            is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f"Trained Model is accepted and model eval artifact is {model_evaluation_artifact}")

                return model_evaluation_artifact 

            # if model is not None : 
            model_list = [model,train_model_object]


            metric_info_artifact = evaluate_regression_model(

                model_list=model_list , 
                X_train=x_train_df , 
                y_train=y_train_arr , 
                X_test=x_test_df , 
                y_test=y_test_arr ,
                base_accuracy= self.model_trainer_artifact.model_accuracy 
            )

            logging.info(f"Model Evaluation is completed and model metric info  artifact :  {metric_info_artifact}")

            if metric_info_artifact is None : 
                response = ModelEvaluationArtifact(is_model_accepted=False,
                                                   evaluated_model_path=trained_model_file_path
                                                   )
                logging.info(response)
                return response 


            if metric_info_artifact.index_number == 1:
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=True)
                self.update_evaluation_report(model_evaluation_artifact)
                logging.info(f"Trined Model is accepted. Model eval artifact {model_evaluation_artifact} created")

            else:
                logging.info("Trained model is not better than existing model ,  hence we are not accepting trained model")
                model_evaluation_artifact = ModelEvaluationArtifact(evaluated_model_path=trained_model_file_path,
                                                                    is_model_accepted=False)
            return model_evaluation_artifact
        except Exception as e : 
            raise TurboException(e,sys)

    def __del__(self):
        logging.info(f"{'=' * 20}Model Evaluation log completed.{'=' * 20} ")




