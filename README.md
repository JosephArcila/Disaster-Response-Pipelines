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

## Data
<img src="https://user-images.githubusercontent.com/39535338/146665047-14eb521c-163d-41ad-9e45-6ee3d665ceda.png" alt="drawing" width="600"/>
The dataset is divided into two csv files
For the data cleaning I combined the messages and categories datasets and merged using the common id, removed duplicates and wrong values.

## Modeling Process
For fitting the data, I tried a RandomForestClassifier, DecisionTreeClassifier, and KNeighborsClassifier, and evaluated precision, recall, f1_score, and accuracy. I prioritized the model that performed best at the f1_macro score since the dataset is imbalanced (ie some labels like water have few examples).
The best model was a DecisionTreeClassifier tunned with various hyperparameters and features.

## Web App
<img src="https://user-images.githubusercontent.com/39535338/146665492-c13a661d-a4ad-45de-a11c-dd9ee868b383.png" alt="drawing" width="800"/>  
<img src="https://user-images.githubusercontent.com/39535338/146665488-0c667944-d98d-43a5-ba56-45a56c6f77e0.png" alt="drawing" width="800"/>  

## Model Results

                             precision    recall  f1-score   support

                   related       0.85      0.86      0.86      5026
                   request       0.58      0.54      0.56      1184
                     offer       0.00      0.00      0.00        32
               aid_related       0.67      0.64      0.66      2819
              medical_help       0.38      0.31      0.34       533
          medical_products       0.46      0.40      0.43       324
         search_and_rescue       0.26      0.19      0.22       180
                  security       0.12      0.08      0.10       122
                  military       0.43      0.38      0.40       221
               child_alone       0.00      0.00      0.00         0
                     water       0.69      0.60      0.64       465
                      food       0.76      0.73      0.74       770
                   shelter       0.63      0.61      0.62       566
                  clothing       0.54      0.51      0.52        99
                     money       0.37      0.29      0.33       170
            missing_people       0.19      0.10      0.13        86
                  refugees       0.34      0.26      0.29       211
                     death       0.60      0.59      0.60       309
                 other_aid       0.32      0.25      0.28       890
    infrastructure_related       0.19      0.16      0.18       413
                 transport       0.34      0.27      0.30       316
                 buildings       0.48      0.41      0.44       361
               electricity       0.37      0.34      0.36       128
                     tools       0.03      0.02      0.03        43
                 hospitals       0.18      0.15      0.17        72
                     shops       0.00      0.00      0.00        23
               aid_centers       0.12      0.08      0.10        75
      other_infrastructure       0.18      0.13      0.15       290
           weather_related       0.73      0.74      0.73      1762
                    floods       0.64      0.58      0.61       549
                     storm       0.65      0.70      0.67       578
                      fire       0.36      0.29      0.32        68
                earthquake       0.79      0.79      0.79       577
                      cold       0.56      0.42      0.48       138
             other_weather       0.22      0.19      0.20       343
             direct_report       0.53      0.46      0.49      1308
             
                 micro avg       0.64      0.60      0.62     21051
                 macro avg       0.40      0.36      0.38     21051
              weighted avg       0.62      0.60      0.61     21051
               samples avg       0.52      0.50      0.46     21051
