'''
Data Preparation for Udacity Disaster Response Project
This file is using three sources to construct a categorization machine
Abdulsalam M. Al-Ali
'''
import sys
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_data(messages_filepath, categories_filepath):
    '''
    loading data from the messages and categories .csv files
    Input: message .csv file path
           categroes .csv file path
    Output: Panda dataframe of the loaded data
    '''
    msg = pd.read_csv(messages_filepath)
    ctg = pd.read_csv(categories_filepath)
    full_df = pd.merge(msg, ctg, on = 'id')
    
    return full_df

def clean_data(df):
    '''
    Input: The loaded dataframe
    Output: The dataframe after cleaning
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)
    # select the first row of the categories dataframe
    row = categories.iloc[1,:]
    # extract the category name
    # I am using a for loop to extract category names
    category_colnames = []
    for i in row:
        category_colnames.append(i[0:-2])
    # Rename the columns of 'categories'
    categories.columns = category_colnames
    
   # convert categroy values to just numbers 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype('float64')

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # removing the indices
    df.reset_index(drop=True, inplace=True)
    categories.reset_index(drop=True, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)

    # Remove Duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessageCategories', engine, index=False) 


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