# Disaster Response Pipeline Project

### Introduction

This repo contains the code required to deploy a Disaster Pipeline MText classifier that contains three main aspects: data cleaning, machine learning model and web app.

### Quick Start Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data_etl/data_processing.py data_source/disaster_messages.csv data_source/disaster_categories.csv data_source/DisasterResponse.db`

    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Todo:

-  check the filepaths are valid before runnung code

### Extract, Transform, Load (ETL)

provide two csv files and location where the db file to be stored

The data source comes in the format of two csv files which are merged and cleaned. This is then outputted to a .db file stored in this folder. The training data is cleaned to provide classification of the messages in to 36 distinct categories.

'Please provide the filepaths of the messages and categories '\
                'datasets as the first and second argument respectively, as '\
                'well as the filepath of the database to save the cleaned data '\
                'to as the third argument. \n\nExample: python process_data.py '\
                'disaster_messages.csv disaster_categories.csv '\
                'DisasterResponse.db'


### Machine Learning

This code allows the user to build a machine learning pipeline using sklearns pipeline functionality (HERE). It is a multioutput classifier.

### Web App

The web app is an interface based on Flask to provide a UI to utilise utilise the machine learning model. It provides some high level visualisations and a UI message classifier. Note the quickstart instructions run the code in debug mode, the live webapp is hosted on Heroku.