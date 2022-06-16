# Disaster Response Pipeline Project
## Installation
This repository was written in HTML and Python , and requires the following Python packages: 
 pandas, numpy, re, pickle, nltk, flask, json, plotly, sklearn, sqlalchemy, sys.
## Project Overview
This code is designed to iniate a  web app which can classify a disaster text messages into several. Users can use this web to determine whether a message contain disaster content or not. From that, appropriate disaster relief agencies can take prompt actions.
The app built to have an ML model to categorize every message received.
## File Description:
There are 3 main folders and 2 files:
1. app
- template
+ master.html # main page of web app
+ go.html # classification result page of web app
- run.py # Flask file that runs app
2. data
- disaster_categories.csv # data to process
- disaster_messages.csv # data to process
- process_data.py # This python excutuble code takes as its input csv files and then creates a SQL database
- DisasterResponse.db # database to save clean data to
3. models
- train_classifier.py # This code trains the ML model with the SQL database
4. ETL Pipeline Preparation.ipynb: process_data.py development procces
5. ML Pipeline Preparation.ipynb: train_classifier.py. development procces
6. README.md
## File Description:
1.Run the following commands in the project's root directory to set up your database and model.
- To run ETL pipeline that cleans data and stores in database 'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'.
- To run ML pipeline that trains classifier and saves 'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
2. Run the following command in the app's directory to run your web app. python run.py




