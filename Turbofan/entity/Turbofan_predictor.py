from Turbofan.exception import TurboException 
from Turbofan.logger import logging 
from Turbofan.util.util import load_data,load_object 
import os , sys 
import pandas as pd 



class TurbofanData: 

    def __init__(self, 
        engineNumber: int,
        cycleNumber: int ,
        sensor2 : float ,
        sensor3 : float ,
        sensor4 : float, 
        sensor7 : float ,
        sensor8 : float, 
        sensor9 : float ,
        sensor11 : float,
        sensor12 : float ,
        sensor13 : float ,
        sensor14 : float ,
        sensor15 : float ,
        sensor17 : float ,
        sensor20 : float ,
        sensor21 : float ,
        RUL : int = None ) :
        try : 
            self.engineNumber = engineNumber,
            self.cycleNumber = cycleNumber ,
            self.sensor2 = sensor2 ,
            self.sensor3 = sensor3 ,
            self.sensor4 = sensor4, 
            self.sensor7 = sensor7,
            self.sensor8 = sensor8, 
            self.sensor9 =sensor9 ,
            self.sensor11 = sensor11,
            self.sensor12 = sensor12 ,
            self.sensor13 = sensor13,
            self.sensor14 = sensor14 ,
            self.sensor15 = sensor15  ,
            self.sensor17 = sensor17 ,
            self.sensor20 = sensor20,
            self.sensor21 = sensor21, 
            self.RUL = RUL
        except Exception as e : 
            raise TurboException(e,sys) from e 

    def get_Turbofan_data_as_dict(self): 
        try : 
            input_data = {
            "Engine Number": self.engineNumber,
            "Current Cycle ": self.cycleNumber ,
            "Total temperature at LPC outlet(°R )" : self.sensor2 ,
           "Total temperature at HPC outlet(°R )" : self.sensor3 ,
            "Total temperature at LPT outlet(°R )" : self.sensor4 , 
            "Total pressure at HPC outlet(psia)" : self.sensor7 ,
            "Physical fan speed(rpm)" : self.sensor8, 
            "Physical core speed (rpm)" : self.sensor9 ,
            "Static pressure at HPC outlet (psia)" : self.sensor11,
            "Ratio of fuel flow to Ps30 (pps/psi)" : self.sensor12 ,
            "Corrected fan speed (rpm)" : self.sensor13,
            "Corrected core speed(rpm)" : self.sensor14 ,
            "Bypass Ratio" : self.sensor15,
            "Bleed Enthalpy" : self.sensor17,
            "HPT coolant bleed(lbm/s )" : self.sensor20 ,
            "LPT coolant bleed (lbm/s)" : self.sensor21  }

            return input_data

        except Exception as e : 
            raise TurboException(e,sys) from e 

    def get_Turbofan_data_dataframe(self): 
        try : 
            Turbofan_data = self.get_Turbofan_data_as_dict() 
            dataframe = pd.DataFrame(Turbofan_data)
            return dataframe
        except Exception as e : 
            raise TurboException(e,sys) from e 



class TurbofanPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise TurboException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise TurboException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            RUL = model.predict(X)
            return RUL
        except Exception as e:
            raise TurboException(e, sys) from e