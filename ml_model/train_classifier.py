import sys
import re
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import time
from sklearn.model_selection import GridSearchCV
import json
import traceback

#!/usr/bin/python

def load_data(database_filepath):

    """ Load processed data from .db file

    Arguments:
        database_filepath {str} -- rel. filepath to .db file
    """
    # create engine and select data from Messages table
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM Messages", engine)

    # Fix for unique values (0, 1, 2), replace 2 as per https://knowledge.udacity.com/questions/70450
    df.related.replace(2, 1, inplace=True)

    X = df['message'] # set messages at X
    Y = df[df.columns[4:]] # set Y as categories
    category_names = df.columns[4:] # rename columns
    
    return X, Y, category_names

# TODO: import 
def tokenize(text):

    """ Clean text to be used in ML algorithm

    Arguments:
        text {str} -- text to be cleaned 
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # normalize text

    words_normalized = text.split() # tokenize text
    
    # Remove stop words
    words_stopw_removed = [word for word in words_normalized if word not in stopwords.words("english")]

    # Reduce words to root form
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words_stopw_removed]

    return words_lemmed


def build_model():

    """ Build ML model using sklearn's pipeline module

    Returns:
        object -- pipeline model
    """

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    return pipeline


def train_model(model, X_train, Y_train):
    return model.fit(X_train, Y_train)


def gridsearch_parameters(pipeline, X_train, Y_train):
    start = time.process_time();

    # parameters = {
    #     'clf__estimator__criterion': ['entropy'], #In this, keep gini or entropy.
    #     'clf__estimator__max_depth': [2, 5], #Use only two
    #     'clf__estimator__n_estimators': [10, 20]# Use only two
    #     #'clf__estimator__min_samples_leaf':[1, 5] # can be ignored
    # }

        # parameters = {
        #     'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        #     'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        #     'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        #     'features__text_pipeline__tfidf__use_idf': (True, False),
        #     'clf__n_estimators': [50, 100, 200],
        #     'clf__min_samples_split': [2, 3, 4],
        #     'features__transformer_weights': (
        #         {'text_pipeline': 1, 'starting_verb': 0.5},
        #         {'text_pipeline': 0.5, 'starting_verb': 1},
        #         {'text_pipeline': 0.8, 'starting_verb': 1},
        #     )
        # }

    parameters = {
    'clf__estimator__n_estimators': [10],
    'vect__ngram_range': [(1, 1)]
}

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, 
                    scoring='f1_macro', cv=None, verbose=10) # , n_jobs=2

    cv.fit(X_train, Y_train)

    print(time.process_time() - start)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """ Evaluate the ML learning model created using pipeline feature in `build_model`

    Arguments:
        model {obj} -- pipeline model to be used in prediction
        X_test {pd.DataFrame} -- single column df containing messages
        Y_test {pd.DataFrame} -- multi column dataframe containing categories data
        category_names {list} -- list of category names from Y_test
    """

    # predict on test data
    Y_pred = model.predict(X_test)
    
    ml_scores = {}
    # display for user/debugging
    for i in range(Y_pred.shape[1]):
        print("Category: " + Y_test.columns[i])
        print(classification_report(Y_test.values[:, i], Y_pred[:, i]))

        ml_scores[Y_test.columns[i]] = classification_report(Y_test.values[:, i], Y_pred[:, i], output_dict=True)

    with open('ml_scores.json', 'w') as fp:
        json.dump(ml_scores, fp)


def save_model(model, model_filepath):
    
    """ Save trained model back as pickle string for use in webapp

    Arguments:
        model {obj} -- model created from ML training
        model_filepath {str} -- where to save the model
    """

    # Save the trained model as a pickle string. 
    pickle.dump(model, open(model_filepath, 'wb'))

def main():

    # check required args given
    if len(sys.argv) == 4:

        database_filepath, model_filepath, optimised_model_filepath = sys.argv[1:]  

        # load data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # TODO: change split size back
        # split data
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.001)

        print('Building pipeline...')
        pipeline = build_model()

        ## Original model
        
        print('Training original model...')
        original_model = train_model(pipeline, X_train, Y_train)

        print('Evaluating original model...')
        evaluate_model(original_model, X_test, Y_test, category_names)

        print('Saving original model...\n    MODEL: {}'.format(model_filepath))
        save_model(original_model, model_filepath)


        # Optimised model

        # try:
        #     print('Optimising model...')
        #     optimised_model = gridsearch_parameters(pipeline, X_train, Y_train)

        #     print('Evaluating optimised model...')
        #     evaluate_model(optimised_model, X_test, Y_test, category_names)

        #     print('Saving optimised model...\n    MODEL: {}'.format(optimised_model_filepath))
        #     save_model(optimised_model, optimised_model_filepath)

        # except Exception:
        #     print("Exception in user code:")
        #     print("-"*60)
        #     traceback.print_exc(file=sys.stdout)
        #     print("-"*60)

        print('Trained optimised model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()