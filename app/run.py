import json
import plotly
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('wordnet')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
# model = joblib.load("../models/DisasterResponse.pkl")
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    """
    OUTPUT:
    Renders the master html template
    """
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)
    
    y_counts = df.iloc[:, 4:].sum().sort_values(ascending=True)
    y_names = list(y_counts.index)
    
    # create visuals
    graphs = [
        
        {
            'data': [
                Bar(
                    x=y_counts,
                    y=y_names,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Categories"
                },
                'xaxis': {
                    'title': "Count"
                },
                'margin': {
                    'l':300,
                    'b':100
                },
                'height': 800,
            }
        },
        
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'margin': {
                    't':100,
                },
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')

def go():
    """
    OUTPUT:
    Renders the go html template, where the user goes after inputing the message
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    OUTPUT:
    a link with the web app hosted locally
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()