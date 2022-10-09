from collections import namedtuple
from Turbofan.component import data_transformation
from Turbofan.component import model_trainer 
from Turbofan.logger import logging ,get_log_file_name
from Turbofan.exception import TurboException 
from datetime import datetime 
import uuid 
from threading import Thread 
from Turbofan.entity.artifact_entity import * 
from Turbofan.entity.config_entity import * 
from Turbofan.component.data_ingestion import * 
from Turbofan.component.data_validation import * 
from Turbofan.component.data_transformation import * 
from Turbofan.component.model_trainer import * 
from Turbofan.component.model_evaluation import * 
from Turbofan.component.model_pusher import * 
from Turbofan.config.configuration import Configuration
import os,sys 
import pandas as pd 
import numpy as np 


Experiment = namedtuple("Experiment", ["experiment_id", "initialization_timestamp", "artifact_time_stamp",
                                       "running_status", "start_time", "stop_time", "execution_time", "message",
                                       "experiment_file_path", "accuracy", "is_model_accepted"])



class Pipeline(Thread): 
    experiment: Experiment = Experiment(*([None] * 11))
    experiment_file_path = None


    def __init__(self,config:Configuration): 
        try : 
            logging.info("Initializing pipeline") 
            os.makedirs(config.training_pipeline_config.artifact_dir ,exist_ok=True)
            Pipeline.experiment_file_path=os.path.join(config.training_pipeline_config.artifact_dir,EXPERIMENT_DIR_NAME, EXPERIMENT_FILE_NAME)
            super().__init__(daemon=False,name='pipeline')
            self.config = config 
        except Exception as e : 
            raise TurboException(e,sys)

    def start_data_ingestion(self)->DataIngestionArtifact: 
        try: 
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config() )
            return data_ingestion.initiate_data_ingestion()
        except Exception as e : 
            raise TurboException(e,sys)

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try : 
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config() , 
                                                data_ingestion_artifact=data_ingestion_artifact) 
            return data_validation.initiate_data_validation()
        except Exception as e : 
            raise TurboException(e,sys)

    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact , 
                                        data_ingestion_artifact:DataIngestionArtifact)->DataTransformationArtifact: 
        try : 
            data_transformation = DataTransformation(data_transformation_config=self.config.get_data_transformation_config(), 
                                                    data_ingestion_artifact=data_ingestion_artifact , 
                                                    data_validation_artifact=data_validation_artifact)
            return data_transformation.initiate_data_transformation()

        except Exception as e : 
            raise TurboException(e,sys)

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact: 
        try : 
            model_trainer = ModelTrainer(model_trainer_config=self.config.get_model_trainer_config(), 
                                            data_transformation_artifact=data_transformation_artifact)
            return model_trainer.initiate_model_trainer() 

        except Exception as  e : 
            TurboException(e,sys)


    def start_model_evaluation(self,data_ingestion_artifact:DataIngestionArtifact , 
                                    data_validation_artifact:DataValidationArtifact, 
                                    model_trainer_artifcat:ModelTrainerArtifact
                                    )->ModelEvaluationArtifact: 
        try :
            model_evaluation = ModelEvaluation(model_evaluation_config=self.config.get_model_evaluation_config() , 
                                            data_ingestion_artifact=data_ingestion_artifact , 
                                            data_validation_artifact=data_validation_artifact, 
                                            model_trainer_artifact=model_trainer_artifcat)
            return model_evaluation.initiate_model_evaluation() 
        except Exception as e : 
            raise TurboException(e,sys)

    def start_model_pusher(self , model_evaluation_artifact:ModelEvaluationArtifact)->ModelPusherArtifact: 
        try : 
            model_pusher = ModelPusher(model_pusher_config=self.config.get_model_pusher_config() , 
                                       model_evaluation_artifact=model_evaluation_artifact) 
            return model_pusher.initiate_model_pusher()
        except Exception as e : 
            raise TurboException(e,sys) 


    def run_pipeline(self): 
        try : 
            if Pipeline.experiment.running_status:
                logging.info("Pipeline is already running")
                return Pipeline.experiment
            # data ingestion
            logging.info("Pipeline starting.")

            experiment_id = str(uuid.uuid4())

            Pipeline.experiment = Experiment(experiment_id=experiment_id,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_time_stamp=self.config.time_stamp,
                                             running_status=True,
                                             start_time=datetime.now(),
                                             stop_time=None,
                                             execution_time=None,
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             is_model_accepted=None,
                                             message="Pipeline has been started.",
                                             accuracy=None,
                                             )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")

            self.save_experiment()

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact
            )
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)

            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    data_validation_artifact=data_validation_artifact,
                                                                    model_trainer_artifcat=model_trainer_artifact)

            if model_evaluation_artifact.is_model_accepted:
                model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
                logging.info(f'Model pusher artifact: {model_pusher_artifact}')
            else:
                logging.info("Trained model rejected.")
            logging.info("Pipeline completed.")

            stop_time = datetime.now()
            Pipeline.experiment = Experiment(experiment_id=Pipeline.experiment.experiment_id,
                                             initialization_timestamp=self.config.time_stamp,
                                             artifact_time_stamp=self.config.time_stamp,
                                             running_status=False,
                                             start_time=Pipeline.experiment.start_time,
                                             stop_time=stop_time,
                                             execution_time=stop_time - Pipeline.experiment.start_time,
                                             message="Pipeline has been completed.",
                                             experiment_file_path=Pipeline.experiment_file_path,
                                             is_model_accepted=model_evaluation_artifact.is_model_accepted,
                                             accuracy=model_trainer_artifact.model_accuracy
                                             )
            logging.info(f"Pipeline experiment: {Pipeline.experiment}")
            self.save_experiment()
        except Exception as e : 
            raise TurboException(e,sys)



    def run(self): 
        try : 
            return self.run_pipeline()  
        except Exception as e : 
            raise TurboException(e,sys)

    def save_experiment(self): 
        try : 
            if Pipeline.experiment.experiment_id is not None:
                experiment = Pipeline.experiment
                experiment_dict = experiment._asdict() ## converting to dictionary 
                experiment_dict: dict = {key: [value] for key, value in experiment_dict.items()}

                experiment_dict.update({
                    "created_time_stamp": [datetime.now()],
                    "experiment_file_path": [os.path.basename(Pipeline.experiment.experiment_file_path)]})

                experiment_report = pd.DataFrame(experiment_dict)

                os.makedirs(os.path.dirname(Pipeline.experiment_file_path), exist_ok=True)
                if os.path.exists(Pipeline.experiment_file_path):
                    experiment_report.to_csv(Pipeline.experiment_file_path, index=False, header=False, mode="a")
                else:
                    experiment_report.to_csv(Pipeline.experiment_file_path, mode="w", index=False, header=True)
            else:
                print("First start experiment")
        except Exception as e : 
            raise TurboException(e,sys)

    @classmethod
    def get_experiments_status(cls, limit: int = 5) -> pd.DataFrame:
        try:
            if os.path.exists(Pipeline.experiment_file_path):
                df = pd.read_csv(Pipeline.experiment_file_path)
                limit = -1 * int(limit)
                return df[limit:].drop(columns=["experiment_file_path", "initialization_timestamp"], axis=1)
            else:
                return pd.DataFrame()
        except Exception as e:
            raise TurboException(e, sys) from e
            
