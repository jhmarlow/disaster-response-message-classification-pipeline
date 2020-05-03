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

def load_model_scores(model_filepath):
    """ Loads classification report data from jsons created in train classifier.py

    Arguments:
        model_filepath {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    # get report automatically created in train_classifier.py
    df = pd.read_json( "../{}.json".format(model_filepath.split('.')[-2]), orient='records')

    names = []
    weighted_avg_f1score = []
    weighted_avg_precision = []
    weighted_avg_recall = []
    macro_avg_f1score = []
    macro_avg_precision = []
    macro_avg_recall = []

    for col_name in df.columns:
        names.append(col_name.replace('_',' '))
        weighted_avg_f1score.append(df[col_name]['weighted avg']['f1-score'])
        weighted_avg_precision.append(df[col_name]['weighted avg']['precision'])
        weighted_avg_recall.append(df[col_name]['weighted avg']['recall'])
        macro_avg_f1score.append(df[col_name]['macro avg']['f1-score'])
        macro_avg_precision.append(df[col_name]['macro avg']['precision'])
        macro_avg_recall.append(df[col_name]['macro avg']['recall'])

    report_df = pd.DataFrame({
        "names": names,
        "weighted_avg_f1score":weighted_avg_f1score,
        "weighted_avg_precision":weighted_avg_precision,
        "weighted_avg_recall":weighted_avg_recall,
        "macro_avg_f1score":macro_avg_f1score,
        "macro_avg_precision":macro_avg_precision,
        "macro_avg_recall":macro_avg_recall
    })

    return report_df


def category_counts(df):
    """ Count of categories in dataset

    Arguments:
        df {[type]} -- [description]
    """
        # Category Counts in training dataset
    category_counts = df[df.columns[4:]].sum()
    categories = df.columns[4:]
    categories = [cat.replace('_', ' ') for cat in categories]
    category_counts_df = pd.DataFrame({'categories':categories, 'category_counts':category_counts})
    category_counts_df = category_counts_df.sort_values(by=['category_counts'], ascending=False)

    return category_counts_df


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


def create_figures(report_df, category_counts_df):
    """ function to contain figures to plotted on webpage

    Arguments:
        report_df {[type]} -- [description]
        category_counts_df {[type]} -- [description]
    """

    # Parameters for plot
    opacity_val = 0.8
    marker_line_color_val = 'rgb(8,48,107)'
    marker_line_width_val = 1.2

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=category_counts_df['categories'],
                    y=category_counts_df['category_counts'],
                    width=0.5,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val
                )
            ],
            'layout': {
                'title': 'Categories in Dataset',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'tickangle':45
                }
            }
        }, 
        {
            'data': [
                Bar(
                    x=report_df["names"],
                    y=report_df["weighted_avg_f1score"],
                    width=0.2,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val,
                    name='Weighted Average F1-score'),
                Bar(
                    x=report_df["names"],
                    y=report_df["weighted_avg_precision"],
                    width=0.2,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val,
                    name='Weighted Average Precision'),
                Bar(
                    x=report_df["names"],
                    y=report_df["weighted_avg_recall"],
                    width=0.2,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val,
                    name='Weighted Average Recall')
            ],
            'layout': {
                'title': 'Machine Learning Classification Report ({})'.format(model_filepath),
                'yaxis': {
                    'title': "Model Score",
                    'range': [0, 1.3]
                },
                'xaxis': {
                    'tickangle':45
                }
            }
        }, 
        {
            'data': [
                Bar(
                    x=report_df["names"],
                    y=report_df["macro_avg_f1score"],
                    width=0.2,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val,
                    name='Macro Average F1-score'),
                Bar(
                    x=report_df["names"],
                    y=report_df["macro_avg_precision"],
                    width=0.2,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val,
                    name='Macro Average Precision'),
                Bar(
                    x=report_df["names"],
                    y=report_df["macro_avg_recall"],
                    width=0.2,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val,
                    name='Macro Average Recall')
            ],
            'layout': {
                'title': 'Machine Learning Classification Report ({})'.format(model_filepath),
                'yaxis': {
                    'title': "Model Score",
                    'range': [0, 1.3]
                },
                'xaxis': {
                    'tickangle':45
                }
            }
        }
        
    ]
    return graphs


def useful_links(classification_results):
    """ Function to return links to certain classification, list in order of priority

    Arguments:
        classification_results {df} -- df describing the classification results, 1 means match

    Returns:
        [str] -- string of link
    """
    if classification_results['medical_help']==1 or classification_results['medical_products']==1:
        link ='https://www.who.int/'
    elif classification_results['earthquake']==1:
        link ='https://earthquake.usgs.gov/earthquakes/map/'
    elif classification_results['weather_related']==1:
        link ='https://www.metoffice.gov.uk/'
    elif classification_results['water']==1:
        link ='https://thewaterproject.org/'
    elif classification_results['search_and_rescue']==1:
        link ='https://www.gov.uk/government/publications/search-and-rescue-framework-uksar' 
    elif classification_results['related']==1 or classification_results['aid_related']==1:
        link ='https://www.redcross.org.uk/'
    else:
        link = None

    return link


# home/index webpage to display data visuals and receive user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ Python code for main page of web application

    Returns:
        obj -- template be rendered in html
    """

    ## Data Manipulation
    # Get model scores from json file created in train classifier.py
    report_df = load_model_scores(model_filepath)
    # count messages per category in database
    category_counts_df = category_counts(df)

    graphs = create_figures(report_df, category_counts_df)
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """ Python code after request for text classification

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
        model_path = "../" + model_filepath # format filepath
        print("Retrieving machine learning model from: " + model_path)
        model = joblib.load(model_path) # load Ml model

        # run app
        print("Running application...")
        main()

    else:
        # ask user to provide required info
        print('\n \nUserInputRequired: Please provide filepaths for data and '
        'Machine Learning Model to be used in this application. \n'
        'e.g. python run.py \'data/data.db\' \'data/model.pkl\' \n \n ')