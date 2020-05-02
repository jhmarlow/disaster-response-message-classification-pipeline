import json
import sys
import plotly
import pandas as pd
import os
import nltk
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib # now required direct call, rather than through sklearn
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """ 
    Tokenize text recieved from user input

    Arguments:
        text {str} -- un-tokenized text
    Returns:
        list -- list of cleaned/tokenized text
    """

    tokens = word_tokenize(text) # split into tokens
    lemmatizer = WordNetLemmatizer() # init

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # clean, to stem, lower case, etc.
        clean_tokens.append(clean_tok)

    return clean_tokens

# home/index webpage to display data visuals and receive user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Python code for main page of web application

    Returns:
        obj -- template be rendered in html
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # genre_counts = df.groupby('genre').count()['message']
    # genre_names = list(genre_counts.index)

    category_counts = df[df.columns[4:]].sum()
    categories = df.columns[4:]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """[summary]

    Returns:
        [type] -- [description]
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results)


def main():
    """ 
    Main function to be called to run flask application
    """

    print("Running app...")
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':

    # check arguments given
    if len(sys.argv) == 3:
            
        database_filepath, model_filepath = sys.argv[1:] # retrieve args

        # load data
        database_path = 'sqlite:///../{}'.format(database_filepath) # format filepath
        print("Retrieving data from: " + database_path)
        engine = create_engine(database_path) # create connection
        df = pd.read_sql_table('Messages', engine) # retrieve data from messages table

        # load model
        model_path = "../ml_model.pkl" # format filepath
        print("Retrieving machine learning model from: " + model_path)
        model = joblib.load(model_path) # load Ml model

        # run app
        main()

    else:
        # ask user to provide required info
        print('\n \nUserInputRequired: Please provide filepaths for data and '
        'Machine Learning Model to be used in this application. \n'
        'e.g. python run.py \'data/data.db\' \'data/model.pkl\' \n \n ')

   