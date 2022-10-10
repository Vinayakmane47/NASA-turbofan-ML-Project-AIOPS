
from flask import Flask , request 
import sys 
import json
import os 
import pandas as pd 
import numpy as np 
from matplotlib.style import context 
from Turbofan.util.util import create_yaml_file , read_yaml_file 
from Turbofan.logger import logging 
from Turbofan.exception import TurboException 
from Turbofan.config.configuration import Configuration 
from Turbofan.constant import get_current_time_stamp , CONFIG_DIR 
from Turbofan.entity.Turbofan_predictor import TurbofanPredictor,TurbofanData 
from Turbofan.pipeline.pipeline import Pipeline 
from flask import send_file, abort, render_template
from Turbofan.logger import get_log_dataframe 

## Defining constants : 
ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "Turbofan_logs"
PIPELINE_FOLDER_NAME = "Turbofan"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)
TURBOFAN_DATA_KEY = "Turbofan_data"
RUL_VALUE_KEY = "RUL"



## Creating Flask Application : 
app = Flask(__name__) 

@app.route('/artifact', defaults={'req_path': 'Turbofan'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("Turbofan", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)
    
    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        TURBOFAN_DATA_KEY: None,
        RUL_VALUE_KEY: None
    }

    if request.method == 'POST':
        engineNumber = float(request.form['engineNumber'])
        cycleNumber =  int(request.form['cycleNumber']) 
        sensor2 = float(request.form['sensor2']) 
        sensor3 = float(request.form['sensor3']) 
        sensor4 = float(request.form['sensor4'])
        sensor7 = float(request.form['sensor7']) 
        sensor8 = float(request.form['sensor8']) 
        sensor9 = float(request.form['sensor9']) 
        sensor11 = float(request.form['sensor11'])
        sensor12  = float(request.form['sensor12']) 
        sensor13 = float(request.form['sensor13']) 
        sensor14 = float(request.form['sensor14']) 
        sensor15 = float(request.form['sensor15']) 
        sensor17 = float(request.form['sensor17']) 
        sensor20 = float(request.form['sensor20']) 
        sensor21 = float(request.form['sensor21']) 

        Turbofan_data = TurbofanData(
            engineNumber = engineNumber,
            cycleNumber = cycleNumber ,
            sensor2 = sensor2 ,
            sensor3 = sensor3 ,
            sensor4 = sensor4, 
            sensor7 = sensor7 ,
            sensor8 = sensor8, 
            sensor9 = sensor9 ,
            sensor11 = sensor11,
            sensor12 = sensor12,
            sensor13 = sensor13 ,
            sensor14 = sensor14 ,
            sensor15 = sensor15 ,
            sensor17 = sensor17,
            sensor20 = sensor20 ,
            sensor21 = sensor21 )

        Turbofan_df = Turbofan_data.get_housing_input_data_frame()
        Turbofan_predictor = TurbofanPredictor(model_dir=MODEL_DIR)

        RUL = Turbofan_predictor.predict(X=Turbofan_df)
        context = {
            TURBOFAN_DATA_KEY: Turbofan_data.get_housing_data_as_dict(),
            RUL_VALUE_KEY: RUL,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            create_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__": 
    app.run()
