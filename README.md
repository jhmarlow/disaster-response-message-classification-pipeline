# Disaster Response Pipeline Project

### Quick Start Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data_etl/data_processing.py data_source/disaster_messages.csv data_source/disaster_categories.csv data_source/disaster_response.db`

    - To run ML pipeline that trains classifier and saves
        `python ml_model/train_classifier.py data_source/disaster_response.db ml_model/optimised_model.pkl`

2. Run the following command in the web_app directory to run your web app.
    `python run.py data_source/disaster_response.db ml_model/optimised_model.pkl`

3. Go to http://0.0.0.0:3001/

![](readme_resources/running_webapp.gif)

4. Enter message, e.g. "I need water" or "there has been an earthquake" - the message will be classified and suitable link provided

### TODO:

- Check the filepaths are valid before running code
- Tackle imbalance problems using downsampling/upsampling or other methods (SMOTE)
- Host app in cloud

### Introduction

This repository contains the code required to deploy a Disaster Response Machine Learning Pipeline. The repository contains three different functionalities:

1. ETL pipeline
2. Machine Learning Text Classifier Model
3. Web App 

The code allows a disaster relief worker to enter a message recieved from different data sources and classify them into 1 of 36 categories present in the dataset. The web app provides some overview statistics on the training dataset, as well as the classification report of the current model being applied. Once the message has been classified it can then be used to suggest a link to a resource.

### 1. Extract, Transform, Load (ETL)

Two .csv files are provided with messages and categories. These are then cleaned and merged into a SQLite database. The training data is cleaned to provide classification of the messages into 36 distinct categories.

### 2. Machine Learning

This code allows the user to build a machine learning pipeline using sklearn's pipeline functionality. As well as optimising the model itself using sklearns Gridsearch functionality.

The Machine Learning Model is a multioutput classifier is scored against the macro F1-score to try and account for class imbalance. That scoring parameters provided are:

- Precision: the ability of the classifier not to label as positive a sample that is negative
- Recall:the ability of the classifier to find all the positive samples
- F1-score is a weighted harmonic mean of the precision and recall. 
- Macro: The 'macro' score calculates the F1 score separated by class but does not use the weights for the aggregation. This results in a bigger penalisation when your model does not perform well with the minority classes.

Reference: https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult

#### NLP

Count vectorizer - using a customized tokenizer
TfidfTransformer - ...

### Web App

The web app is an interface based on Python's Flask to provide a UI to  utilise the machine learning model. It provides some high level visualisations and a UI message classifier. Note the quickstart instructions run the code in debug mode, the live webapp is hosted on Heroku.

![](readme_resources/webapp_demo.gif)

### Dataset Discussion

The dataset displays class imbalance, meaning that in some cases there are very few cases in the training dataset where messages for a particular class can be provided (e.g. water, see "Support" in classification reports). As there are very few true positives in the training set it makes it difficult for the ML model to learn how to classify these categories. This can be imporved by emphasizing class and recall for particular categories #TODO:.....