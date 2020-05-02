# Disaster Response Pipeline Project

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

-  Check the filepaths are valid before running code


### Introduction

This repo contains the code required to deploy a Disaster Pipeline Machine Learning Text Classifier. The repository contains three main processes: data cleaning, machine learning model and web app deployment. The code allows a disaster relief worker to enter a message recieved from anumerous sources and classify them into 1 of 36 categories. The web app provides some overview statistics on the training dataset, as well as teh classification report of the current model being applied.

### Extract, Transform, Load (ETL)

Two .csv files are provided with messages and categories. These are then cleaned and merged into a SQlite database. The training data is cleaned to provide classification of the messages in to 36 distinct categories.
Splits the string into different features to be classified in the Machine Learning model.

### Machine Learning

This code allows the user to build a machine learning pipeline using sklearn's pipeline functionality (HERE). As well as optimising the model itself using the Gridsearch functionality. It is a multioutput classifier. 

The Machine Learning Model is then scored against:

- Precision is the ability of the classifier not to label as positive a sample that is negative
- Recall is the ability of the classifier to find all the positive samples
Recall = TP/(TP+FN) and precision = TP/(TP+FP)
- The F-measure can be interpreted as a weighted harmonic mean of the precision and recall. A measure reaches its best value at 1 and its worst score at 0. With F1, recall and the precision are equally important.

            F1 = 2 * (precision * recall) / (precision + recall)

- Weighted: The weighted score calculates the F1 score for each class independently and then adds them together using a weight that depends on the number of true labels of each class - therefore favouring the majority class.

            𝐹1𝑐𝑙𝑎𝑠𝑠1∗𝑊1 + 𝐹1𝑐𝑙𝑎𝑠𝑠2∗𝑊2 + ⋅⋅⋅ + 𝐹1𝑐𝑙𝑎𝑠𝑠𝑁∗𝑊𝑁

- Macro: The 'macro' score calculates the F1 score separated by class but does not use the weights for the aggregation. This results in a bigger penalisation when your model does not perform well with the minority classes.

            𝐹1𝑐𝑙𝑎𝑠𝑠1 + 𝐹1𝑐𝑙𝑎𝑠𝑠2 + ⋅⋅⋅ + 𝐹1𝑐𝑙𝑎𝑠𝑠𝑁

If you are worried with class imbalance I would suggest using 'macro'. However, it might be also worthwile implementing some of the techniques available to taclke imbalance problems such as downsampling the majority class, upsampling the minority, SMOTE, etc.

Reference: https://datascience.stackexchange.com/questions/40900/whats-the-difference-between-sklearn-f1-score-micro-and-weighted-for-a-mult


### Web App

The web app is an interface based on Flask to provide a UI to utilise utilise the machine learning model. It provides some high level visualisations and a UI message classifier. Note the quickstart instructions run the code in debug mode, the live webapp is hosted on Heroku.

### Data set

This dataset is imbalanced (ie some labels like water have few examples). In your README, discuss how this imbalance, how that affects training the model, and your thoughts about emphasizing precision or recall for the various categories.