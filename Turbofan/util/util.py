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
            
