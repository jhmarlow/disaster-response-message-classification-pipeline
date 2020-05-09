"""Train classifer.

Returns:
    [type] -- [description]
"""

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
import sys
import re
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# !/usr/bin/python


def load_data(database_filepath):

    """Load processed data from .db file.

    Arguments:
        database_filepath {str} -- rel. filepath to .db file
    """
    # create engine and select data from Messages table
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM Messages", engine)

    # Fix for unique values (0, 1, 2), replace 2 as per
    # https://knowledge.udacity.com/questions/70450
    df.related.replace(2, 1, inplace=True)

    X = df['message']  # set messages at X
    Y = df[df.columns[4:]]  # set Y as categories
    category_names = df.columns[4:]  # rename columns
    return X, Y, category_names


def tokenize(text):

    """Clean text to be used in ML algorithm.

    Arguments:
        text {str} -- text to be cleaned
    """

    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())  # normalize text

    words_normalized = text.split()  # tokenize text
    # Remove stop words
    words_stopw_removed = [word for word in words_normalized
                            if word not in stopwords.words("english")]

    # Reduce words to root form
    words_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words_stopw_removed]

    return words_lemmed


def build_model():
    """Build ML model using sklearn's pipeline module.

    Returns:
        object -- pipeline model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    return pipeline


def gridsearch_parameters(pipeline, X_train, Y_train):
    """Function used to optimized ml model based on hyperparameters.

    Arguments:
        pipeline {obj} -- pipeline setup for ml model
        X_train {df} -- messages
        Y_train {df} -- categories

    Returns:
        obj -- trained ml model
    """
    start = time.process_time()

    parameters = {
        'vect__ngram_range': [[1, 1], [1, 2]],
        'tfidf__use_idf': [True, False]}

# 'clf__estimator__criterion': ['entropy'], #In this, keep gini or entropy.
# 'clf__estimator__max_depth': [2, 5], # Use only two
# 'clf__estimator__n_estimators': [20, 50], # Use only two
# 'clf__estimator__min_samples_leaf':[1, 5], # can be ignored

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
                    # scoring='f1_macro', cv=None, verbose=10)   # , n_jobs=2

    cv.fit(X_train, Y_train)

    print("Model optimising time.... " + str(time.process_time() - start))
    return cv


def evaluate_model(model, X_test, Y_test, category_names, model_filepath):
    """Evaluate the ML learning model created using pipeline feature
    in `build_model`.

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

    scores_filepath = model_filepath.split('.')[-2] + ".json"
    with open(scores_filepath, 'w') as fp:
        json.dump(ml_scores, fp)


def save_model(model, model_filepath):
    """Save trained model back as pickle string for use in webapp.

    Arguments:
        model {obj} -- model created from ML training
        model_filepath {str} -- where to save the model
    """
    # Save the trained model as a pickle string
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Call function called to train classifier."""
    # check required args given
    if len(sys.argv) == 3:

        database_filepath, optimised_model_filepath = sys.argv[1:]

        # load data
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # split data
        X_train, X_test, Y_train, Y_test = \
            train_test_split(X, Y, test_size=0.3)

        print('Building pipeline...')
        pipeline = build_model()

        print('Optimising model...')
        optimised_model = gridsearch_parameters(pipeline, X_train, Y_train)

        print('Evaluating optimised model...')
        evaluate_model(optimised_model, X_test, Y_test,
                       category_names, optimised_model_filepath)

        print('Saving optimised model...\n'
              'MODEL: {}'.format(optimised_model_filepath))
        save_model(optimised_model, optimised_model_filepath)

        print('Trained optimised model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
