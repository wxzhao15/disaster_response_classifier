# Disaster Response Pipeline Project

## Project File Structure
### Folders and Files 
#### data 
storing raw data files, data pipeline scripts as well as the transformed data files
1. raw data
    - categories.csv
    - messages.csv
2. database
    - database_disaster_response.db
3. script
    - process_data.py (main process to clean and transform raw data into analyzble format)
    - 'ETL Pipeline Preparation.ipynb' (draft script which later be cleaned and organized into process_data.py)

#### models
containing the machine learning pipeline script for disaster response classfication model building and fine-tuning
1. script
    - train_classifier.py (main process to build machine learning pipeline)
    - 'ML Piepeline Preparation.ipynb' (draft script which later be cleaned and organized into train_classifier.py)
2. saved models (ignored in git process due to their large size)
    - pickle format models being trained, fine-tuned, and saved from the ML process

#### app
containing the pyton files that create the front-end interface which hosts the ML model. Provide platform for users to visualize the results of the model
    - run.py (python script building the app lnterface)

## Packages used in this project
- numpy
- pandas
- sqlalchemy
- nltk
- sklearn
- re
- pickle


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/database_disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/database_disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
