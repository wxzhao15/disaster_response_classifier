# Disaster Response Pipeline Project

### Project File Structure
## folders and files included in this repository 
# data 
storing raw data files, data pipeline scripts as well as the transformed data files

# models
containing the machine learning pipeline script for disaster response classfication model building and fine-tuning

# app
containing the pyton files that create the front-end interface which hosts the ML model. Provide platform for users to visualize the results of the model


### Packages used in this project
- nltk
- sklearn
- re
- pickle


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/database_disaster_response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/database_disaster_response.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
