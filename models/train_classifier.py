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
    # connect to SQL database
    engine = create_engine('sqlite:///../data/database_disaster_response.db')
    # load data
    df = pd.read_sql_table("msg_data.db", con = engine)

    # extract X, Y, and category names to be predicted
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = list(Y.columns)

    return X, Y, category_names


def tokenize(text):
    # normalizing text to lower case and remove punctuations
    normalize_text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower())
    words = word_tokenize(normalize_text)

    # remove stop words and lemmatize 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]

    return tokens


def build_model():
    ## using the best params found by GridSearch in part below
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(sublinear_tf=True)),
        ('cls',MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
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