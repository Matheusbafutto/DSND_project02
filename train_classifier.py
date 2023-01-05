from datetime import datetime
import sys
import pickle
import os
from time import time
import pandas as pd
from sqlalchemy import create_engine

from nltk import word_tokenize, download
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

download("punkt")
download("wordnet")

"""
    load data from SQLite file and outputs features, category outputs and category labels
    @param database_filepath: file path to input SQLite file to be used for loading the data
    @returns X: pandas Series of raw tweet text messages
    @returns Y: pandas DataFrame of category classifications
    @returns labels: array of labels for classifications on Y
"""
def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql(sql='SELECT * FROM Tweets', con=engine)
    X = df.message.values
    Y = df.loc[:, df.columns[4:]]
    return X, Y, Y.columns

"""
    custom tokenization function for normalizing and lemmatizing text messages
    @param text: text message to be tokenized
    @returns tokenized vector
"""
def tokenize(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text.lower())

    new_token_list = []
    for token in tokens:
        lemmatized_token = lemmatizer.lemmatize(token)
        new_token_list.append(lemmatized_token)
    return new_token_list

"""
    build_model: method to output model to be used in ML pipeline
    @returns sklearn estimator
"""
def build_model():
    # build estimator to be used in grid search
    n_cpus = os.cpu_count()
    pipeline = Pipeline([
        ('vec', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_jobs=n_cpus - 1, n_estimators=200)) # leave 1 cpu to handle non training processes
    ])
    return pipeline

"""
    evaluate_model: computes precision, recall and f1 score metrics for each classification label and prints it
    @param model: sklearn estimator to be evaluated
    @param X_test: feature vector from ML pipeline test dataset
    @param Y_test: true 36 label classifications from test dataset
    @param category_names: label names for each category on Y_test
"""
def evaluate_model(model, X_test, Y_test, category_names):
    def display_results(y_pred, y_test):
        results = {}
        for label in y_test.columns:
            results[label] = [
                precision_score(y_pred=y_pred[label], y_true=y_test[label]),
                recall_score(y_pred=y_pred[label], y_true=y_test[label]),
                f1_score(y_pred=y_pred[label], y_true=y_test[label]),
            ]
        print(pd.DataFrame(results, index=["precision", "recall", "f1_score"]).T.sort_values(by="f1_score"))

    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    display_results(Y_pred, Y_test)

"""
    save_model: saves sklearn model into a pickle file
    @param model: generic sklearn estimator
    @param model_filepath: file path the pickle file should be store into
"""
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

"""
    measure_execution_time: utility method to display time of execution of a function call
    @param function: a generic callable method
    @param *args: any other args the callable may need to execute properly
"""
def measure_execution_time(function, *args):
    start = datetime.now()
    function(*args)
    end = datetime.now()
    print(f"[{function.__name__}] execution time: {end - start}")

"""
    main: entry point for this ML pipeline script
"""
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        measure_execution_time(model.fit, X_train, Y_train)

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