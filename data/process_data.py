# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys

def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
    messages_filepath - csv file with the emergency messages
    categories_filepath - csv file with the labels of each emergency message
    OUTPUT:
    df - DataFrame: Combined dataset with the messages and categories datasets merged using the common id
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories)
    return df

def clean_data(df):
    """
    INPUT:
    df - DataFrame: Combined dataset with the messages and categories datasets merged using the common id
    OUTPUT:
    df - Cleaned DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # use this row to extract a list of new column names for categories
    category_colnames = [cat.split('-')[0] for cat in row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [row[1] for row in categories[column].str.split('-')]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories, left_index=True, right_index=True)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop rows with wrong values
    inv_idx = df[(df.iloc[:, 4:] > 1).any(1)].index
    df.drop(index=inv_idx, inplace=True)
    return df

def save_data(df, database_filename):
    """
    INPUT:
    df - Cleaned DataFrame
    database_filename - Desired name for the databese
    OUTPUT:
    stores the cleaned DataFrame in a SQLite database
    """
    # Save the clean dataset into an sqlite database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace')

def main():
    """
    OUTPUT:
    Load, clean and save the data 
    """
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