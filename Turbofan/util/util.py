from ast import Str
import yaml 
import os,sys
from Turbofan.exception import TurboException 
import Turbofan.logger 
import dill 
import pandas as pd 
import numpy as np 
from Turbofan.constant import *



def create_yaml_file(file_path:str , data:dict): 
    """ 
        This Function create yaml file at specified location
    """
    try : 
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'w') as yaml_file :
            if data is not None : 
                yaml.dump(data,yaml_file) 
    except Exception as e : 
        raise TurboException(e,sys)


def read_yaml_file(file_path:str): 

    """ 
      This Function will read the yaml files 
    """
    
    try : 
        with open(file_path , "rb") as file : 
            return yaml.safe_load(file)
    except Exception as e : 
        raise TurboException(e,sys)
            

def save_numpy_array(file_path: str , array: np.array): 

    try : 
        dir_name = os.path.dirname(file_path) 
        os.makedirs(dir_name,exist_ok=True) 
        with open(file_path,"wb") as file: 
            np.save(file,array)

    except Exception as e : 
        raise TurboException(e,sys)

def load_numpy_array(file_path:str): 
    try : 
        with open(file_path,"rb") as file : 
            return np.load(file) 
    except Exception as e : 
        raise TurboException(e,sys)

def save_object(file_path:str , obj): 
    try : 
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as file : 
            dill.dump(obj,file)

    except Exception as e : 
        raise TurboException(e,sys)

def load_object(file_path:str): 
    try : 
        with open(file_path,"rb") as file_obj : 
            return dill.load(file_obj) 

    except Exception as e : 
        raise TurboException(e,sys)

def load_data(file_path: str, schema_file_path: str) -> pd.DataFrame:
    try:
        datatset_schema = read_yaml_file(schema_file_path)

        schema = datatset_schema[DATASET_SCHEMA_COLUMNS_KEY]

        dataframe = pd.read_csv(file_path)

        error_messgae = ""


        for column in dataframe.columns:
            if column in list(schema.keys()):
                dataframe[column].astype(schema[column]) # for dtype casting 
            else:
                error_messgae = f"{error_messgae} \nColumn: [{column}] is not in the schema."
        if len(error_messgae) > 0:
            raise Exception(error_messgae)
        return dataframe

    except Exception as e:
        raise TurboException(e,sys) from e

