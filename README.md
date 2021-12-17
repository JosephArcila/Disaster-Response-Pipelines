# Disaster Response Pipeline Project

## Summary
Analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages. This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.


## How to run the Python scripts and web app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Files

### ETL Pipeline Preparation.ipynb
Loads the messages and categories datasets -> Merges the two datasets -> Cleans the data -> Stores it in a SQLite database

### ML Pipeline Preparation.ipynb
Loads data from the SQLite database -> Splits the dataset into training and test sets -> Builds a text processing and machine learning pipeline -> Trains and tunes a model using GridSearchCV -> Outputs results on the test set -> Exports the final model as a pickle file

### app
- template
    - master.html: Main page of web app.
    - go.html: Classification result page of web app.
- run.py: Flask file that runs app

### data
- disaster_categories.csv: Data to process. 
- disaster_messages.csv: A data set containing real messages that were sent during disaster events.
- process_data.py: A data pipeline to prepare message data from major natural disasters around the world.
- InsertDatabaseName.db: Database to save clean data to.

### models
- train_classifier.py: A machine learning pipeline to categorize emergency messages based on the needs communicated by the sender so that you can send the messages to an appropriate disaster relief agency.
- classifier.pkl: Saved model 
