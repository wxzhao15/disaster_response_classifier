# import libraries
import sys
import nltk
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

nltk.download(['stopwords','wordnet','punkt','omw-1.4'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    
    """
    Loading data from database - database_disaster_db with the defined database_filepath

    Parameteres: 
    - database_filepath (str): The path to sqlite database data file

    Output:
    - X (pandas.Series): Independent variables containing messages data
    - Y (pandas.DataFrame): Dependent variables containing labeled categories
    - category_names: a list of category names for prediction
    """

    # connect to SQL database
    engine = create_engine('sqlite:///data/database_disaster_response.db')
    # load data
    df = pd.read_sql_table(database_filepath, con = engine)

    # extract X, Y, and category names to be predicted
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):

    """
    Transforming text to tokens through steps:
    - lowring case
    - removing stop words
    - lemmatizing 
    - tokenizing

    Parameters:
    - text (str): message data

    Returns:
    - tokens (list): cleaned and tokenized message data

    """

    # normalizing text to lower case and remove punctuations
    normalize_text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    words = word_tokenize(normalize_text)

    # remove stop words and lemmatize 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return tokens


def build_model():
    """
    Build machine learning pipeline for text classification using 
    CountVectorizer and TfidfTransformer from sklearn package

    In the trianing process we have fine-tuned the pipeline with GridSearch.
    This Pipeline preprocess textual data with vectorization and Tfidf transformation
    and using a multi-output classifier (with RF method) to predict on 36 labels classification

    Returns:
    - pipeline (sklearn.pipeline.Pipeline): A machine learning pipeline for text
      classification.
    """

    ## using the best params found by GridSearch in part below
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('cls',MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluating model performance using test dataset

    Parameters:
    - model: loaded text classification model
    - X_test, Y_test: test dataset with preprocessed dependent and independent data
    - category_names: the list of category names for prediction

    Output:
    this function will print out a pandas dataframe with column names "f1", "precision", "recall" 
    row index being each predicting category that model classify for 

    """


    Y_pred = model.predict(X_test)

    f1= []
    precision= []
    recall = []

    for col in range(Y_test.shape[1]):
        
        col_name = Y_test.columns[col]
        y_test = Y_test.loc[:, col_name].to_list()
        y_pred = list(Y_pred[:, col])

        report = classification_report(y_test,y_pred, output_dict = True)

        f1.append(report['macro avg']['f1-score'])
        precision.append(report['macro avg']['precision'])
        recall.append(report['macro avg']['recall'])
        # f1[col_name] = report['macro avg']['f1-score']
        # precision[col_name] = report['macro avg']['precision']
        # recall[col_name] = report['macro avg']['recall']

    results_dict = {'f1':f1, 'precision': precision, 'recall': recall}
    results_df = pd.DataFrame(results_dict, index = category_names)

    print(results_df)


def save_model(model, model_filepath):
    """
    Saving model to the defined model_filepath

    Parameters:
    - model (a fitted Multioutput Classifier)
    - model_filepath (str ending with .pkl): defining filepath to save the fitted classifier 
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


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