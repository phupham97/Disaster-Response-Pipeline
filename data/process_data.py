import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function load the data into a dataframe.
    Input: paths to csv file which contains data.
    Output: Dataframe of data.
    '''
    #Load Messages Dataset
    messages = pd.read_csv(messages_filepath)
    
    #Load Categories Dataset
    categories = pd.read_csv(categories_filepath)
    
    #Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])
    
    return df



def clean_data(df):
     '''
    This function clean dataframe.
    Input: a dataframe.
    Output: Dataframe after cleaning.
    '''
    #Create a df for category 
    categories = df['categories'].str.split(';', expand = True)
    #Select first row
    row_1 = categories.iloc[0]
    # Extract a list of categories names.
    category_colnames = []
    for i in row_1:
        category_colnames.append(i[:-2])
    print(category_colnames)
    # Rename the columns of `categories`
    categories.columns = category_colnames
    # Convert  category numbers
    for column in categories:
        # set every value to be the last character of a string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # Drop column which has only value 0
    categories.drop('child_alone', axis = 1, inplace = True)
    # Drop the  categories  from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # Concatenate   dataframe with the new `categories` 
    df = pd.concat([df, categories], axis = 1)
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    # Remove rows with a  value of 2 from df
    df = df[df['related'] != 2]
    
    return df



def save_data(df, database_filename):
    """Save into  SQLite database.
    
    inputs:
    df: dataframe. Dataframe containing cleaned version of merged message and 
    categories data.
    database_filename: string. Filename for output database.
       
    outputs:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages & categories '\
              'datasets as the first & second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
