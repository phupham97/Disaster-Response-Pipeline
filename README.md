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

