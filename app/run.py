import json
import plotly
import pandas as pd

from nltk import download
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
app = Flask(__name__)

download("punkt")
download("wordnet")

"""
    tokenization function to be used in the vectorization of new Tweets
    @param text string containing the message to be tokenized
    @returns vector of normalized and lemmatized words from tweet
"""
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('Tweets', engine)

# load model
model = joblib.load('./models/vec_tfidf_RandomForest.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    #chart 2
    class_names = list(filter(lambda col: col not in set(['id', 'message', 'original', 'genre']), df.columns))
    class_counts = df[class_names].sum().sort_values(ascending=False)


    # chart 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = [
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
                }
            }
        },
        {
            'data': [
                Bar(
                    y=class_counts,
                    x=[name.replace("_", " ").capitalize() for name in class_counts.index],
                    # orientation='h'
                )
            ],
            'layout': {
                'title': 'Class occurences per class',
                'yaxis': {
                    'title': "Count"
                    'rotate'
                },
                'height': 1000,
                'xaxis': {
                    'title': "Class"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()