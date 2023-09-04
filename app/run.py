import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/database_disaster_response.db')
df = pd.read_sql_table('msg_data.db', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals

    ## genre summary
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ## extract pct of label categorized summary
    categories_df = df.drop(['id','message','original','genre'], axis=1)
    label_df = (categories_df != 0).mean()

    label_pct = list(label_df)
    categories_names = list(label_df.index)
    print(categories_names)

    ## extract in each genre what's the pct of the msg related to aid
    genre_aid_df = df.groupby('genre').agg({
        'id':'size',
        'aid_related':'sum'
    }).reset_index()
    genre_aid_df['pct'] = genre_aid_df['aid_related']/genre_aid_df['id']
    genre_aid_value = genre_aid_df['pct']
    genre_aid_type = genre_aid_df['genre']

    ## extract in each genre what's the pct of the msg related
    genre_related_df = df.groupby('genre').agg({
        'id':'size',
        'related':'sum'
    }).reset_index()
    genre_related_df['pct'] = genre_related_df['related']/genre_related_df['id']
    genre_related_value = genre_related_df['pct']
    genre_related_type = genre_related_df['genre']

    ## extract in each genre what's the pct of the msg related
    genre_request_df = df.groupby('genre').agg({
        'id':'size',
        'request':'sum'
    }).reset_index()
    genre_request_df['pct'] = genre_request_df['request']/genre_request_df['id']
    genre_request_value = genre_request_df['pct']
    genre_request_type = genre_request_df['genre']

    # create visuals
    graphs = [
        
        {
            'data': [
                Bar(
                    x = categories_names,
                    y = label_pct
                )
            ],

            'layout': {
                'title': 'Aid and Weather Are Most Appeared Categories Among All Response Messages Related',
                'xaxis': {
                    'title': 'Label Category'
                    },
                'yaxis': {
                    'title': 'Pct of messages not related to label'
                    }
            }
        },
        {
            'data': [
                Bar(
                    x = genre_related_type,
                    y = genre_related_value
                )
            ],

            'layout': {
                'title': 'Social Media Messages Have Highest Mix Related to Disaster',
                'xaxis': {
                    'title': 'Message Genre Types'
                    },
                'yaxis': {
                    'title': 'Pct of messages related'
                    }
            }

        },

        {
            'data': [
                Bar(
                    x = genre_aid_type,
                    y = genre_aid_value
                )
            ],

            'layout': {
                'title': 'News messages have highest percentage related to aid, while social media messages being lowest',
                'xaxis': {
                    'title': 'Message Genre Types'
                    },
                'yaxis': {
                    'title': 'Pct of messages related to aid'
                    }
            }

        },


        {
            'data': [
                Bar(
                    x = genre_request_type,
                    y = genre_request_value
                )
            ],

            'layout': {
                'title': 'Direct messages have the highest percentage related to request',
                'xaxis': {
                    'title': 'Message Genre Types'
                    },
                'yaxis': {
                    'title': 'Pct of messages related to request'
                    }
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
                }
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