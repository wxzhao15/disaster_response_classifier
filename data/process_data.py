import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Loading raw messages and categories data from data folder using read_csv 
    and merge two datasets together joining on 'id' (with inner join to remove
    data rows that without corresponding label or message information)

    Parameters:
    - messages_filepath(str): file path to messages data
    - categories_filepath(str): file path to categories data

    Return:
    - df (pandas.DataFrame): merged dataframe
    """

    # read in data with the input file paths
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge two datasets
    df=messages.merge(categories, how = 'inner', on = 'id')

    return df


def clean_data(df):
    """
    cleaning data and transform them into analyzable format. In this function, we 
    split categories into 36 clean separate columns with corresponding category name

    Parameters:
    df (pandas.DataFrame): dataframe with both raw message and category information

    Output: 
    df (pandas.DataFrame): cleaned dataframe with categories being splited into 36
    sperate columns and duplicated rows removed
    """

    # split categories into seperate columns
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = [col[:-2] for col in row]
    
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)

    # remove duplicated rows
    df.drop_duplicates(inplace=True)
    return(df)


def save_data(df, database_filename):
    """
    saving cleaned data frame into database with the database_filename 
    defined in argument

    parameters
    df (pandas.DataFrame): cleaned dataframe to be loaded in the sql database
    database_filename (str): file name for the datatable to be saved as in defined database
    """
    
    engine = create_engine('sqlite:///data/database_disaster_response.db')
    df.to_sql(database_filename, engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()