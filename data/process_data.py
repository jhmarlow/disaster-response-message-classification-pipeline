import sys
import pandas as pd
from sqlalchemy import create_engine

class ProcessData:

    def __init__(self):
        pass

    def load_data(self, messages_filepath, categories_filepath):

        """ A function to load and combine two .csv files provided.
        Arguments:
            messages_filepath {str} -- file path to messages .csv
            categories_filepath {str} -- filepath to categories .csv
        Returns:
            pd.DataFrame -- df containing csvs merged on id
        """

        messages = pd.read_csv(messages_filepath) # load messages
        categories = pd.read_csv(categories_filepath) # load categories

        return messages.merge(categories, left_on='id', right_on='id') # merge two dfs on id

    def clean_categories_data(self, df):
        """ Split single string into 36 individual category columns
        Arguments:
            df {pd.DataFrame} -- original dataframe
        Returns:
            pd.DataFrame -- df with categories split into columns
        """
        # 
        return df['categories'].str.split(';', expand=True)

    def rename_category_columns(self, categories):
        """ Selects the first row of the categories dataframe and
        extracts a list of new column names for categories.
        Arguments:
            categories {pd.DataFrame} -- original categories df with categories split into cols
        Returns:
            pd.DataFrame -- returns categories dataframe with renamed columns
        """
        categories.columns = categories.iloc[0].apply(lambda x: x[:-2])
        return categories

    def convert_string_numbers(self, categories):
        """converts string number values into int, using the assumption that the last
        character will represent the correct int value
        Arguments:
            categories {pd.DataFrame} -- categories df split into columns and renamed
        Returns:
            pd.DataFrame -- with values represented as int
        """
        # Convert category values to just numbers 0 or 1
        categories['offer'].apply(lambda x: x[-1:]).astype(int)

        for column in categories:
        # set each value to be the last character of the string
        # convert column from string to numeric
            categories[column] = categories[column].apply(lambda x: x[-1:]).astype(int)

        return categories

    def merge_remove_duplicates(self, df, categories):
        """Join original dataframe with cleaned categories df
        Arguments:
            df {pd.DataFrame} -- original df joined from the 2 csv files
            categories {pd.DataFrame} -- cleaned categories df
        Returns:
            pd.DataFrame -- joined clean df 
        """
        # drop the original categories column from `df`
        df = df.drop(labels='categories', axis=1)

        # concatenate the original dataframe with the new `categories` dataframe
        df = pd.concat([df, categories], axis=1)

        # drop duplicates
        df = df.drop_duplicates();

        return df

    def clean_data(self, df):
        """[summary]

        Arguments:
            df {pd.DataFrame} -- original datafrm

        Returns:
            pd.DataFrame -- cleaned dataframe
        """

        categories = self.clean_categories_data(df)
        categories = self.rename_category_columns(categories)
        categories = self.convert_string_numbers(categories)
        df = self.merge_remove_duplicates(df, categories)

        return df

    def save_data(self, df, database_path):
        """ Saves cleaned data back into .db file
        Arguments:
            df {pd.DataFrame} -- cleaned data
            database_path {str} -- complete relative file path of where file to be saved 
        """
        engine = create_engine('sqlite:///{}'.format(database_path))
        df.to_sql('Messages', engine, index=False)  


    def main(self):
        """ 
        Main method performing ETL tasks and providing user feedback
        """

        # check arguments given
        if len(sys.argv) == 4:
            
            messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

            # Start processing
            print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
                .format(messages_filepath, categories_filepath))
            df = self.load_data(messages_filepath, categories_filepath) # retrieve data from files

            print('Cleaning data...')
            df = self.clean_data(df) # clean data for ML
            
            print('Saving data...\n    DATABASE: {}'.format(database_filepath))
            self.save_data(df, database_filepath) # save file back to .db
            
            print('Cleaned data saved to database!')
        
        else:
            print('Please provide the filepaths of the messages and categories '\
                'datasets as the first and second argument respectively, as '\
                'well as the filepath of the database to save the cleaned data '\
                'to as the third argument. \n\nExample: python process_data.py '\
                'disaster_messages.csv disaster_categories.csv '\
                'DisasterResponse.db')


if __name__ == '__main__':
    ProcessData().main()