from Turbofan.logger import logging
from Turbofan.exception import TurboException 
from Turbofan.entity.config_entity import DataIngestionConfig, DataValidationConfig 
from Turbofan.entity.artifact_entity import DataIngestionArtifact , DataValidationArtifact 
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
import os,sys 
import pandas as pd 
import json 



class DataValidation: 

    def __init__(self , data_validation_config:DataValidationConfig, 
                        data_ingestion_artifact:DataIngestionArtifact) -> None:
        try : 
            logging.info(f"{'=='*20} Data Validatioon Started {'=='*20}")
            self.data_validation_config = data_validation_config 
            self.data_ingestion_artifact = data_ingestion_artifact 

        except Exception as e  :
            raise TurboException(e,sys)

    def get_train_test_df(self): 
        try : 
            logging.info("Getting train test dataframe") 
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            return train_df , test_df
        except Exception as e : 
            raise TurboException(e,sys)

    def is_train_test_file_exists(self): 
        try :
            logging.info("Checking training and test file exists or not ")
            is_train_file_path = False 
            is_test_file_path = False 

            train_file_path = self.data_ingestion_artifact.train_file_path 
            test_file_path = self.data_ingestion_artifact.test_file_path 

            is_train_file_path = os.path.exists(train_file_path)
            is_test_file_path = os.path.exists(test_file_path)

            available = is_train_file_path and is_test_file_path 

            if not available : 
                train_file_path = self.data_ingestion_artifact.train_file_path 
                test_file_path = self.data_ingestion_artifact.test_file_path 

                message = f"Training file {train_file_path} or test file {test_file_path} is not present in directory"
                raise Exception(message)
            return available

        except Exception as e : 
            raise TurboException(e,sys)

    def get_and_save_data_drift_report(self): 
        try :
            profile = Profile(sections=[DataDriftProfileSection()])

            train_df,test_df = self.get_train_test_df()

            profile.calculate(train_df,test_df)

            report = json.loads(profile.json())

            report_file_path = self.data_validation_config.report_file_path
            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir,exist_ok=True)

            with open(report_file_path,"w") as report_file:
                json.dump(report, report_file, indent=6)
            return report 
        except Exception as e : 
            raise TurboException(e,sys)

    def save_data_drift_report_page(self): 
        try : 
            dashboard = Dashboard(tabs=[DataDriftTab()])
            train_df,test_df = self.get_train_test_df()
            dashboard.calculate(train_df,test_df)

            report_page_file_path = self.data_validation_config.report_page_file_path
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)

        except Exception as e : 
            raise TurboException(e,sys)

    def is_data_drift_found(self): 
        try : 
            report = self.get_and_save_data_drift_report()
            self.save_data_drift_report_page()
            return True 
        except Exception as e : 
            raise TurboException(e,sys)

    def initiate_data_validation(self) -> DataValidationArtifact: 
        try : 
            self.is_train_test_file_exists()
            self.is_data_drift_found()
            data_validation_artifact = DataValidationArtifact( 
                schema_file_path=self.data_validation_config.schema_file_path , 
                report_file_path= self.data_validation_config.report_file_path , 
                report_page_file_path=self.data_validation_config.report_page_file_path ,
                is_validated=True , 
                message = "Data Validation succesfull"
            )
            logging.info(f"Data Validation artifact {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e : 
            raise TurboException(e,sys)


