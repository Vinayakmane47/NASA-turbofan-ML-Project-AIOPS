import os 
from datetime import datetime 

def get_current_time_stamp(): 
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
