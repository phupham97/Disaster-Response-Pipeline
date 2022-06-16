# Import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('wordnet') 
nltk.download('stopwords')
nltk.download('punkt')

def load_data(database_filepath):
    '''
    This function load the data for training.
    Input: paths to SQL database.
    Output: 
    X: messages column,
    Y: output column,
    category_names: name for categories columns.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # listing the columns
    category_names = list(np.array(y.columns))

    return X, y, category_names

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_model():
    pipe = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__max_features': ['auto', 'sqrt'],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__min_samples_leaf':[1, 2, 5]}

    cv = GridSearchCV(pipe, param_grid=parameters)

    return cv

# Define function to calculate accuracy:
def class_report(Array1, Array2, col_names):
    '''
    This function evaluates accuracy between 2 arrays.
    Input: 
    Array1: output array,
    Array2: prediction array,
    col_names: categories name.
    Output: 
    data_report: A report evaluates precision, recall, f1 score of each categories.
    '''
    class_report = []
    # Evaluate metrics for each set of labels
    for i in range(len(col_names)):
        precision = classification_report(Array1[:,i], Array2[:,i]).split()[-4]
        recall = classification_report(Array1[:,i], Array2[:,i]).split()[-3]
        f1 = classification_report(Array1[:,i], Array2[:,i]).split()[-2]
        class_report.append([precision, recall, f1])
    # Store metrics
    class_report = np.array(class_report)
    data_report = pd.DataFrame(data = class_report, index = col_names, columns = ['Precision', 'Recall', 'F1'])
      
    return data_report

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates accuracy of the model.
    Input: 
    Model: model which trained,
    X_test: test set of categories column,
    Y_test: test set of output column,
    category_names: categories name.
    Output: 
    A report evaluates accuracy of the model.
    '''
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    print(class_report(np.array(Y_test), y_pred, col_names=category_names))
    
def save_model(model, model_filepath):
    """
    Save model to a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
