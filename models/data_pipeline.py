# import packages
import sys
import pickle

from sklearn.model_selection import train_test_split 


def load_data(data_file):
    # read in file


    # clean data


    # load to database


    # define features and label arrays


    return X, y


def build_model():
    # text processing and model pipeline


    # define parameters for GridSearchCV


    # create gridsearch object and return as final model pipeline


    return model_pipeline

def train(X, y, model):
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # fit model
    model = build_model()
    model.fit(X_train, y_train)

    # output model test results
    y_pred = model.predict(X_test)

    return model


def export_model(model):
    # Export model as a pickle file
    with open('RF_multioutput_model_final.pkl', 'wb') as f:
        pickle.dump(model, f)


def run_pipeline(data_file):
    X, y = load_data(data_file)  # run ETL pipeline
    model = build_model()  # build model pipeline
    model = train(X, y, model)  # train model pipeline
    export_model(model)  # save model


if __name__ == '__main__':
    data_file = sys.argv[1]  # get filename of dataset
    run_pipeline(data_file)  # run data pipeline