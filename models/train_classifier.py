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
 
def load_data(database_filepath):
    """Load processed data from .db file
    Arguments:
        database_filepath {str} -- rel. filepath to .db file
    """

    # create engine and select data from Messages table
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM Messages", engine)

    # Fix for unique values (0, 1, 2), replace 2 as per https://knowledge.udacity.com/questions/70450
    df.related.replace(2, 1, inplace=True)

    X = df['message']
    Y = df[df.columns[4:]]
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    """Clean text to be used in ML algorithm

    Arguments:
        text {str} -- text to be cleaned 
    """

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize text
    words_normalized = text.split()
    
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

    # display for user/debugging
    for i in range(Y_pred.shape[1]):
        print("Category: " + Y_test.columns[i])
        print(classification_report(Y_test.values[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    """ Save trained model back as pickle string for use in webapp

    Arguments:
        model {obj} -- model created from ML training
        model_filepath {str} -- where to save the model
    """

    # Save the trained model as a pickle string. 
    pickle.dump(model, open('ml_model.pkl', 'wb')) 

def main():

    # check required args given
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        
        # Train model
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()