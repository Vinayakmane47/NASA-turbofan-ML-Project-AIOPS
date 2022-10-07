from Turbofan.entity.config_entity import DataIngestionConfig
from Turbofan.logger import * 
from Turbofan.constant import * 
import os,sys 
from Turbofan.exception import TurboException 
from Turbofan.entity.artifact_entity import DataIngestionArtifact 
import pandas as pd 
import numpy as np 
from six.moves import urllib 

class DataIngestion: 

    def __init__(self, data_ingestion_config:DataIngestionConfig) -> None:
        try : 
            logging.info(f"{'=='*20}Data Ingestion started{'=='*20}") 
            self.data_ingestion_config = data_ingestion_config 

        except Exception as e : 
            raise TurboException(e,sys)

    def download_turbofan_data(self): 
        try :
            logging.info(f"Downloading Turbofan Dataset ") 
            download_url_train = self.data_ingestion_config.download_url_train
            download_url_test = self.data_ingestion_config.download_url_test 
            
            ## Getting dirs : 
            ingested_train_dir = self.data_ingestion_config.ingested_train_dir
            ingested_test_dir = self.data_ingestion_config.ingested_test_dir 
            file_name_train = os.path.basename(download_url_train)
            file_name_test = os.path.basename(download_url_test)

            ## getting train , test dir 
            train_file_path = os.path.join(ingested_train_dir,file_name_train)
            test_file_path = os.path.join(ingested_test_dir,file_name_test)

            os.makedirs(ingested_train_dir,exist_ok=True)
            os.makedirs(ingested_test_dir,exist_ok=True)
            ## Downloading an saving train and test files : 
            urllib.request.urlretrieve(download_url_train,train_file_path) # train 
            urllib.request.urlretrieve(download_url_test,test_file_path)  # test

            data_ingestion_artifact = DataIngestionArtifact(train_file_path=train_file_path ,test_file_path=test_file_path,is_ingested=True , 
                                        message="Data Ingestion Succesfull Completed")
            logging.info(f"Data Ingestion artifact info {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e : 
            raise TurboException(e,sys)

    def initiate_data_ingestion(self)-> DataIngestionArtifact: 
        try : 
            data_ingestion_artifact = self.download_turbofan_data() 
            return data_ingestion_artifact
        except Exception as e : 
            raise TurboException(e,sys)

    
            




