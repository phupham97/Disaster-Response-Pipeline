# Disaster Response Pipeline Project

## Project Overview
This code is designed to iniate a  web app which can classify a disaster text messages into several.
This repository was written in HTML and Python , and requires the following Python packages: 
 pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys.

The app built to have an ML model to categorize every message received.
## File Description:
* **process_data.py**: This python excutuble code takes as its input csv files and then creates a SQL database
* **train_classifier.py**: This code trains the ML model with the SQL database
* **ETL Pipeline Preparation.ipynb**:  process_data.py development procces
* **ML Pipeline Preparation.ipynb**: train_classifier.py. development procces
* **data**: This folder contains sample messages and categories datasets in csv format.
* **app**: cointains the run.py to iniate the web app.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
   Or Go to http://localhost:3001/
