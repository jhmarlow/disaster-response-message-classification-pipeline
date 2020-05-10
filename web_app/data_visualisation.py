"""Contains code for webapp visualisations."""

import pandas as pd
from plotly.graph_objs import Bar


def load_model_scores(model_filepath):
    """Load classification report data from jsons created in train classifier.py.

    Arguments:
        model_filepath {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # get report automatically created in train_classifier.py
    df = pd.read_json("../{}.json".format(model_filepath.split('.')[-2]),
                      orient='records')

    names = []
    weighted_avg_f1score = []
    weighted_avg_precision = []
    weighted_avg_recall = []
    macro_avg_f1score = []
    macro_avg_precision = []
    macro_avg_recall = []

    for col_name in df.columns:
        names.append(col_name.replace('_', ' '))
        weighted_avg_f1score.append(df[col_name]['weighted avg']['f1-score'])
        weighted_avg_precision.append(
            df[col_name]['weighted avg']['precision'])
        weighted_avg_recall.append(df[col_name]['weighted avg']['recall'])
        macro_avg_f1score.append(df[col_name]['macro avg']['f1-score'])
        macro_avg_precision.append(df[col_name]['macro avg']['precision'])
        macro_avg_recall.append(df[col_name]['macro avg']['recall'])

    report_df = pd.DataFrame({
        "names": names,
        "weighted_avg_f1score": weighted_avg_f1score,
        "weighted_avg_precision": weighted_avg_precision,
        "weighted_avg_recall": weighted_avg_recall,
        "macro_avg_f1score": macro_avg_f1score,
        "macro_avg_precision": macro_avg_precision,
        "macro_avg_recall": macro_avg_recall
    })

    return report_df


def category_counts(df):
    """Count categories in dataset.

    Arguments:
        df {[type]} -- [description]
    """
    # Category Counts in training dataset
    category_counts = df[df.columns[4:]].sum()
    categories = df.columns[4:]
    categories = [cat.replace('_', ' ') for cat in categories]
    category_counts_df = pd.DataFrame({'categories': categories,
                                       'category_counts': category_counts})
    category_counts_df = category_counts_df.sort_values(by=['category_counts'],
                                                        ascending=False)

    return category_counts_df


def create_figures(report_df, category_counts_df, df, model_filepath):
    """Plot figures for webpage.

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
                    'tickangle': 45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=df.groupby('genre').count()['message'].index,
                    y=df.groupby('genre').count()['message'].values,
                    width=0.5,
                    opacity=opacity_val,
                    marker_line_color=marker_line_color_val,
                    marker_line_width=marker_line_width_val
                )
            ],
            'layout': {
                'title': 'Genres in Dataset',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'tickangle': 45
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
                'title': 'Machine Learning Classification Report ({})'
                         .format(model_filepath),
                'yaxis': {
                    'title': "Model Score",
                    'range': [0, 1.3]},
                'xaxis': {
                    'tickangle': 45}
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
                'title': 'Machine Learning Classification Report ({})'
                         .format(model_filepath),
                'yaxis': {
                    'title': "Model Score",
                    'range': [0, 1.3]},
                'xaxis': {
                    'tickangle': 45}
            }
        }
    ]

    return graphs
