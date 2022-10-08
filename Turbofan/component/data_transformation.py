from Turbofan.exception import TurboException 
from Turbofan.logger import logging 
from Turbofan.config.configuration import DataTransformationConfig 
from Turbofan.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact
from Turbofan.constant import * 
import pandas as pd 
import numpy as np 
import os,sys 
from Turbofan.util.util import * 
from sklearn.preprocessing import StandardScaler , PolynomialFeatures 
from sklearn.feature_selection import SelectFromModel 
from sklearn.svm import SVR


class DataTransformation: 


    def __init__(self , data_transformation_config:DataTransformationConfig , 
                        data_ingestion_artifact:DataIngestionArtifact , 
                        data_validation_artifact:DataValidationArtifact) -> None:



        try : 
            logging.info(f"{'=='*20} Data Transformation Started {'=='*20}")
            self.data_transformation_config = data_transformation_config 
            self.data_ingestion_artifact = data_ingestion_artifact 
            self.data_validation_artifact = data_validation_artifact 

        except Exception as e : 
            raise TurboException(e,sys)


    def initiate_data_transformation(self): 
        try : 
            train_file_path = self.data_ingestion_artifact.train_file_path 
            test_file_path = self.data_ingestion_artifact.test_file_path 

            schema_file_path = self.data_validation_artifact.schema_file_path 

            logging.info(f"Loading dataset as train and test dataframe") 

            train_df = load_data(file_path=train_file_path , schema_file_path=schema_file_path)
            test_df = load_data(file_path=test_file_path , schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)
            target_column = schema[TARGET_COLUMN_KEY]

            logging.info(f"splitting training and test dataset into x and y features i.e. dependent and independent")
            x_train_df = train_df.drop(columns=[target_column],axis=1)
            y_train = train_df[target_column]

            x_test_df = test_df.drop(columns=[target_column],axis=1)
            y_test = test_df[target_column]

            logging.info("performing Data Transformation ")

            logging.info("Standard Scaling")
            scalar = StandardScaler()
            x_train_arr = scalar.fit_transform(x_train_df)
            x_test_arr = scalar.transform(x_test_df)

            logging.info("Adding Polynomial Features")
            poly = PolynomialFeatures(2)
            x_train_arr = poly.fit_transform(x_train_arr)
            x_test_arr = poly.fit_transform(x_test_arr)

            logging.info("getting important features from polynomial features")
            
            preprocessed_obj_path = self.data_transformation_config.preprocessed_object_file_path

            if os.path.exists(preprocessed_obj_path): 
                svr = load_object(file_path=preprocessed_obj_path)
            else : 
                svr = SVR(kernel='linear')
                svr.fit(x_train_arr,y_train) 
                save_object(file_path=preprocessed_obj_path,obj=svr)


            select_features = SelectFromModel(svr, threshold='mean', prefit=True) 
            select_features.get_support()
            feature_names = poly.get_feature_names()
            logging.info(f"Important features are {np.array(feature_names)[select_features.get_support()]}")
            x_train_arr = x_train_arr[:,select_features.get_support()]
            x_test_arr = x_test_arr[:,select_features.get_support()]

            logging.info("concating x and y arrays")

            train_arr = np.c_[ x_train_arr, np.array(y_train)]
            test_arr = np.c_[ x_test_arr, np.array(y_test)]

            transformed_train_dir = self.data_transformation_config.transformed_train_dir 
            transformed_test_dir = self.data_transformation_config.transformed_test_dir 

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info("saving transformed train and test files") 
            save_numpy_array(file_path=transformed_train_file_path,array=train_arr) # train 
            save_numpy_array(file_path=transformed_test_file_path,array=test_arr) # test

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=svr)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessed_object_file_path=preprocessing_obj_file_path

            )
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e : 
            raise TurboException(e,sys)


    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")


