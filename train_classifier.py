'''
In this file, a machine learning model will be prepared from the created .db file from the previous process_data.py file
Abdulsalam M. Al-Ali
'''
# import libraries
import sys
import re
import numpy as np
import pandas as pd
import pickle

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


def load_data(database_filepath):
    '''
    Input: database file path
    Output: Loaded data in a panda dataframe
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('MessageCategories', engine)
    X = df['message'] # choosing the feature column
    Y = df.iloc[:,4:40] # choosing the output columns
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    '''
    This function takes a text and tokenize it to be used in the ML model
    Other cleaning are done to the text, such as lowering the cases and removing stop words     and punctuations
    '''
    # lowering the case (normalization)
    new_text = text.lower()
    # removing punctuation
    new_text = re.sub(r"[^a-zA-Z0-9]", " ", new_text)
    # tokenization
    new_text = word_tokenize(new_text)
    # removing stop words
    new_text = [w for w in new_text if w not in stopwords.words("english")]
    # lemmatization
    # Reduce words to their root form
    new_text = [WordNetLemmatizer().lemmatize(w) for w in new_text]
    # Lemmatize verbs by specifying pos
    new_text = [WordNetLemmatizer().lemmatize(w, pos='v') for w in new_text]


    return new_text


def build_model():
    '''
    In this function, a ML model will be built to find the best categories for a given         disaster message   
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
    'clf__estimator__max_depth' : [5,40,None],
    'clf__estimator__min_samples_leaf' : [2,5,10]
    }

    cv = GridSearchCV(pipeline, parameters, n_jobs = -2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    ''' Evaluating the model and printing its results
    Input: model
           X_test: testing features
           y_test: testing outcomes
           category_name: list of output (categories) names
    '''
    
    y_pred = model.predict(X_test)
    
    # printing the results
    for i in range(len(category_names)):
        print('category name: ', category_names[i],'\n', classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))

def save_model(model, model_filepath):
    ''' 
    Saving the model .pkl file to be used in other apps
    Input: model: the fitted model
           model_filepath: where to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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