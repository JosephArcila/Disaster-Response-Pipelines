import sys
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    INPUT:
    database_filepath - sqlite database file path
    OUTPUT:
    X - DataFrame with the text input data for the machine learning model
    Y - DataFrame with the labels of the of the messages that will be used to train the model
    category_names - list with the names of each emmergency category
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(database_filepath, con=engine)
    X = df.loc[:, 'message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    INPUT:
    text - String to be tokenized
    OUTPUT:
    tokenized - list of words tokenized, case normalized, lemmatized and stemmed
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Split text into words using NLTK
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v').strip() for w in lemmed]
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in lemmed]
    tokenized = stemmed
    return tokenized


def build_model():
    """
    OUTPUT:
    cv - model pipeline from the gridsearch object
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))
    ])
                           
    parameters = {
        'clf__estimator__criterion': ['gini','entropy']
    }

    scoring = {"Accuracy": "accuracy", "Precision": make_scorer(precision_score, average='macro', zero_division=0), 
           "Recall": make_scorer(recall_score, average='macro', zero_division=0), 
           "F1": make_scorer(f1_score, average='macro', zero_division=0)}
                           
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, cv=2, return_train_score=True, refit='F1', scoring = scoring))

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    INPUT:
    model - trained model
    X_test - Test set DataFrame with the text input data for the machine learning model
    Y_test - DataFrame with the labels to evaluate the model
    category_names - List of the different possible categories of the multi output classifier
    OUTPUT:
    classification report - Text summary of the precision, recall, F1 score for each class.
    """
    best_model = model.best_estimator_
    y_pred = best_model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    INPUT:
    model - trained model
    model_filepath - file path name where to store the model
    OUTPUT:
    export model as a pickle file
    """
    joblib.dump(model, model_filepath)


def main():
    """
    OUTPUT:
    Exports a model to a pickle file that uses the message column to predict classifications for 36 categories
    """
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