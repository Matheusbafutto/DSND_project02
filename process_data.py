import sys
import re
import pandas as pd
from sqlalchemy import create_engine

"""
    Merges csv files for messages and categories into one pandas DataFrame (merge assumes common ids)
    @param messages_filepath: filepath to a csv dataset containing messages from tweets
    @param categories_filepath: filepath to a csv dataset containing categories from tweets
    @returns merged datasets as a pandas DataFrame
"""
def load_data(messages_filepath, categories_filepath):
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    df = messages.merge(right=categories, on='id', how='inner')
    return df

"""
    cleans up merged tweets dataset
    @param df: pandas DataFrame containing tweet messages and categories
    @returns clean, deduplicated DataFrame with tidy categories column
"""
def clean_data(df):
    # tidy and clean categories column
    categories = df.categories.str.split(pat=';', expand=True)
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda value: re.sub(pattern=r'^(\w+)-\d+', repl='\\1', string=value))
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # replace dirty categories with new clean one
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories],axis=1)

    # handle duplicates by dropping them
    df.drop_duplicates(inplace=True)

    # drop unknown values in related column
    df = df.drop(index=df.query('related == 2').index)

    return df

"""
    saves dataset as a SQLite database file
    @param df: pandas DataFrame containing tweet messages and categories
    @param database_filename: file path for output SQLite file to be saved on
    @returns nothing
"""
def save_data(df, database_filename):
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Tweets', engine, index=False, if_exists='replace')

"""
    main: entry point of ETL script
"""
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