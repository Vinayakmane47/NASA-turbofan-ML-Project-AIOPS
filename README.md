
# NASA Turbofan Engine RUL Prediction
### Application URL : https://nasa-turbofan-ml-project.herokuapp.com/
### Demo Link - https://youtu.be/WK7NbTI4pNk


## Introduction : 
Turbofan engine is a highly complex and precise thermal machinery, 
which is the “heart” of the aircraft.
About 60% of the total faults of the aircraft are related to 
the turbofan engine. The prognosis of the remaining useful 
life (RUL) of turbofan engine provides an important basis for 
predictive maintenance and remanufacturing, and plays a major
role in reducing failure rate and maintenance cost. RUL is the 
total number of flights remained for the engine after the last 
flight. In this project i have used Machine Learning approach 
to predict RUL. This project is able to train different models 
and choose the best one by evaluating accuracy scores of each 
model





![Logo](https://evolution.skf.com/wp-content/uploads/sites/5/2016/11/16-4-aerospace-fig-5-en.jpg)


## Problem Statement :

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure.The objective  is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data

## Dataset: 
Datasets include simulations of multiple turbofan engines over time, each row contains the following information:
1. Engine unit number
2. cycles
3. Three operational settings
4. 21 sensor readings 





## Resources : 

 - [Turbofan Engine](https://en.wikipedia.org/wiki/Turbofan)
 - [NASA Turbofan Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
 


## Steps : 

1. Create Github repo with `readme.md` ,`.gitignore`. 
2. Clone this repo in local system and start coding in `VS-Code`. 
 
```bash
  $ git clone 
  $ code .
```
3. Create Conda Enviornment 
```bash
  $ conda create -p venv python == 3.7 -y 
  $ conda activate venv/ 

```
4. create `requirements.txt` , `app.py ` and `setup.py` file. 
5. Create Heroku application. 
6. Setup CI-CD pipeline : 
  - create `Dockerfile` 
  - create `.dockerignore` file 
  - create `.github\workflow` folders and add `main.yaml` file
7. For CI-CD pipeline we need to add Enviornmental variables in Github 
   The following variables can be added in github secrets : 
- `HEROKU_API_KEY`
- `HEROKU_APP_NAME`
- `HEROKU_EMAIL`
Now type this commands : 
```bash
  $ git add . 
  $ git commit -m "first commit" 
  $ git push origin main 
```
Deployement is completed via github. 

8. Create Folder structure like this in `VS-CODE` : 
```bash

.github
|     └─── workflows
|           |── main.yaml
|
└─── config
|     |── config.yaml
|     |── model.yaml
      |── schema.yaml
|
└─── Turbofan
|     |──  __init__.py
|     |── component ───|
|     |                |── __init__.py
|     |                |── data_ingestion.py
|     |                |── data_validation.py
|     |                |── data_transformation.py
|     |                |── model_evaluation.py
|     |                |── model_pusher.py
|     |                |── model_trainer.py
|     |
|     |── config
|     |── constant
|     |── entity
|     |── exception
|     |── logger
|     |── pipeline
|     |── util
|     
|
└─── templates
|     |──  experiment_history.html
|     |── files.html
|     |── header.html
|     |── index.html
|     |── log_files.html
|     |── predict.html
|     |── saved_model_file.html
|     |── train.html
|     |── update_model.html
|
|── .dockerignore
|── .gitignore
|── app.py
|── requirements.txt
|── Dockerfile
|── setup.py
```

9. Dockerfile contains Following content: 
```bash
  FROM python:3.7
  COPY . /app
  WORKDIR /app
  RUN pip install -r requirements.txt
  EXPOSE $PORT
  CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app

```

10. Start writing each module and package. 

## MLOPS Pipeline Overview : 
![Logo](https://blogs.nvidia.com/wp-content/uploads/2020/09/1-MLOps-NVIDIA-invert-final.jpg)



## Models Used for Training : 
- SVR 
- KNN 
- XGBOOST 
- RANDOM FOREST 
- NAIVE BAYES 
- DECISION TREE
- 
## Some Features of project : 

- Able to train different models.
- Able to choose best parameters of models using grid search CV
- Compare all trained models using `r2_score` and model accuracy. 
- Select the best model which have best accuracy as well as best parameters. 
- Compare accuracy score with models which are present in history an select best one. 




## Result  and Conclusion :

We found out that SVR is the best fitted model for RUL prediction with model accuracy of 67.5%. Train dataset accuracy is around 71 %  which is quite good. Knn Model can also able to give good results but it is having less accuracy than SVR. We are able to deploy the project in heroku by implementing CI-CD pipeline.  

![RUL_IMG](https://user-images.githubusercontent.com/103372852/195010802-29a4b5e2-40dd-4a74-83f8-9623663b7d10.png)









    
    
