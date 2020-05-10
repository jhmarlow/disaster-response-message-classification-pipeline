"""Code to run to start webapp."""

from data_visualisation import category_counts,\
    load_model_scores, create_figures
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request  # , jsonify
from sqlalchemy import create_engine
import json
import sys
import plotly
import pandas as pd
import joblib # now required direct call, rather than through sklearn
import nltk
nltk.download('punkt')


app = Flask(__name__)


def tokenize(text):
    """Tokenize text recieved from user input.

    Arguments:
        text {str} -- un-tokenized text
    Returns:
        list -- list of cleaned/tokenized text
    """
    tokens = word_tokenize(text)  # split into tokens
    lemmatizer = WordNetLemmatizer()  # init

    clean_tokens = []
    for tok in tokens:
        # clean, to stem, lower case, etc.
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def useful_links(classification_results):
    """Return links in order of priority for classification.

    Arguments:
        classification_results {df} -- df describing the classification
        results, 1 means match

    Returns:
        [str] -- string of link
    """
    if classification_results['medical_help'] == 1 or \
       classification_results['medical_products'] == 1:
        link = 'https://www.who.int/'
    elif classification_results['earthquake'] == 1:
        link = 'https://earthquake.usgs.gov/earthquakes/map/'
    elif classification_results['weather_related'] == 1:
        link = 'https://www.metoffice.gov.uk/'
    elif classification_results['water'] == 1:
        link = 'https://thewaterproject.org/'
    elif classification_results['search_and_rescue'] == 1:
        link = 'https://www.gov.uk/government/publications/search-and-rescue-framework-uksar'
    elif classification_results['related'] == 1 or \
        classification_results['aid_related'] == 1:
        link = 'https://www.redcross.org.uk/'
    else:
        link = None

    return link


# home/index webpage to display data visuals and receive user input for model
@app.route('/')
@app.route('/index')
def index():
    """Python code for main page of web application.

    Returns:
        obj -- template be rendered in html
    """
    # Data Manipulation
    # Get model scores from json file created in train classifier.py
    report_df = load_model_scores(model_filepath)
    # count messages per category in database
    category_counts_df = category_counts(df)

    graphs = create_figures(report_df, category_counts_df, df, model_filepath)

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Python code after request for text classification.

    Returns:
        [obj] -- html template to render
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    useful_link = useful_links(classification_results)

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        useful_link=useful_link,
        classification_result=classification_results)


def main():
    """Call to run flask application."""
    print("Running app...")
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':

    # check arguments given
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]  # retrieve args

        # load data
        # format filepath
        database_path = 'sqlite:///../{}'.format(database_filepath)
        print("Retrieving data from: " + database_path)
        engine = create_engine(database_path)  # create connection
        # retrieve data from messages table
        df = pd.read_sql_table('Messages', engine)

        # load model
        model_path = "../" + model_filepath  # format filepath
        print("Retrieving machine learning model from: " + model_path)
        model = joblib.load(model_path)  # load Ml model

        # run app
        print("Running application...")
        main()

    else:
        # ask user to provide required info
        print('\n \nUserInputRequired: Please provide filepaths for data and '
              'Machine Learning Model to be used in this application. \n'
              'e.g. python run.py \'data/data.db\' \'data/model.pkl\' \n \n ')
