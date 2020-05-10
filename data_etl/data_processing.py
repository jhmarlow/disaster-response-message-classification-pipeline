"""File to hold all the data processing code for disaster pipeline."""

import sys
import pandas as pd
from sqlalchemy import create_engine


class ProcessData:
    """Process data class."""

    def __init__(self):
        """init."""
        pass

    def load_data(self, messages_filepath, categories_filepath):
        """Load and combine two .csv files provided on id column.

        Arguments:
            messages_filepath {str} -- file path to messages .csv
            categories_filepath {str} -- filepath to categories .csv
        Returns:
            pd.DataFrame -- df containing csvs merged on id
        """
        messages = pd.read_csv(messages_filepath)  # load messages
        categories = pd.read_csv(categories_filepath)  # load categories
        
        # merge the two dfs on id
        return messages.merge(categories, left_on='id', right_on='id')

    def split_categories_data(self, df):
        """Split single string into 36 individual category columns.

        Arguments:
            df {pd.DataFrame} -- original dataframe
        Returns:
            pd.DataFrame -- df with categories split into columns
        """
        # split categories column string
        return df['categories'].str.split(';', expand=True)

    def rename_category_columns(self, categories):
        """Select the first row of the categories dataframe and
        extracts a list of new column names for categories.

        Arguments:
            categories {pd.DataFrame} -- original categories df with
            categories split into cols
        Returns:
            pd.DataFrame -- returns categories dataframe with renamed columns
        """
        # get the category names
        categories.columns = categories.iloc[0].apply(lambda x: x[:-2])
        return categories

    def convert_string_numbers(self, categories):
        """Convert string number values into int.

        Using the assumption that the last
        character will represent the correct int value.

        Arguments:
            categories {pd.DataFrame} -- categories df split into columns and
            renamed
        Returns:
            pd.DataFrame -- with values represented as int
        """
        # Convert category values to just numbers 0 or 1
        categories['offer'].apply(lambda x: x[-1:]).astype(int)

        # set each value to be the last character of the string
        # convert column from string to numeric
        for column in categories:
            categories[column] = \
                categories[column].apply(lambda x: x[-1:]).astype(int)

        return categories

    def merge_remove_duplicates(self, df, categories):
        """Join original dataframe with cleaned categories df.

        Arguments:
            df {pd.DataFrame} -- original df joined from the 2 csv files
            categories {pd.DataFrame} -- cleaned categories df
        Returns:
            pd.DataFrame -- joined clean df
        """
        # drop the original categories column from `df`
        df = df.drop(labels='categories', axis=1)
        # concatenate the original dataframe with the new `categories`
        # dataframe
        df = pd.concat([df, categories], axis=1)

        return df.drop_duplicates()  # drop duplicates

    def clean_data(self, df):
        """Use previously defined methods to clean data.

        Arguments:
            df {pd.DataFrame} -- original datafrm

        Returns:
            pd.DataFrame -- cleaned dataframe
        """
        categories = self.split_categories_data(df)  # split categories data
        categories = self.rename_category_columns(categories)  # rename columns
        # convert string to int
        categories = self.convert_string_numbers(categories)
        # remove any duplicates
        return self.merge_remove_duplicates(df, categories)

    def save_data(self, df, database_path):
        """Save cleaned data back into .db file.

        Arguments:
            df {pd.DataFrame} -- cleaned data
            database_path {str} -- complete relative file path of where file
            to be saved
        """
        engine = create_engine('sqlite:///{}'.format(database_path))
        df.to_sql('Messages', engine, index=False, if_exists='replace')

    def main(self):
        """Perform ETL tasks and providing user feedback."""
        if len(sys.argv) == 4:  # check number of arguments given

            # declare argument variables
            messages_filepath, categories_filepath, database_filepath \
                = sys.argv[1:]

            # Start processing

            print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
                  .format(messages_filepath, categories_filepath))
            # retrieve data from files
            df = self.load_data(messages_filepath, categories_filepath)

            print('Cleaning data...')
            df = self.clean_data(df)  # clean data for ML

            print('Saving data...\n    DATABASE: {}'.format(database_filepath))
            self.save_data(df, database_filepath)  # save file back to .db

            print('Cleaned data saved to... ' + str(database_filepath))

        else:
            print('Please provide the filepaths of the messages and categories'
                  'datasets as the first and second argument respectively, as '
                  'well as where to save the cleaned data '
                  'to as the third argument. \n\nExample: python'
                  'data_etl/data_processing.py '
                  'data_source/disaster_messages.csv'
                  'data_source/disaster_categories.csv '
                  'data_source/DisasterResponse.db')


if __name__ == '__main__':
    # init class and call main to run ETL tool
    ProcessData().main()
