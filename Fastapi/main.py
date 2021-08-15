# pip install numpy joblib pandas matplotlib seaborn sklearn nltk surprise jinja2 python-multipart aiofiles
import numpy as np
import joblib
import numpy as np
from pandas.core.frame import DataFrame
from fastapi import FastAPI, Form,Request
from fastapi.templating import Jinja2Templates
import json

# prediction

# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
import pickle
from fastapi.staticfiles import StaticFiles
import warnings; warnings.simplefilter('ignore')

app = FastAPI()

# model = joblib.load(open('model.pkl', 'rb'))

# cv = joblib.load(open('cv.pkl', 'rb'))

templates = Jinja2Templates(directory="templates/")

app.mount("/static", StaticFiles(directory="static/"), name="static")

# Tfid Vector - Description Predict
smd = pd. read_csv('./datasets/tfid_smd.csv')
tf1 = pickle.load(open("./models/tfidf1.pkl", 'rb'))
cosine_sim = linear_kernel(tf1, tf1)
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])

# Count vector - Meta Data Predict
count_smd = pd. read_csv('./datasets/count_smd.csv')
count1 = pickle.load(open("./models/count1.pkl", 'rb'))
cosine_sim_1 = cosine_similarity(count1, count1)
count_smd = count_smd.reset_index()
titles_1 = count_smd['title']
indices_1 = pd.Series(count_smd.index, index=count_smd['title'])


def get_recommendations(title,indices,titles: DataFrame,cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    # print(movie_indices)
    # print(titles.iloc[2182])
    return list(titles.iloc[movie_indices])

@app.get('/')
def home(request: Request):
    prediction_text = ""
    return templates.TemplateResponse("Movie.html",context={'request': request, 'prediction_text': prediction_text})

# Movie Dexcription Based prediction
@app.post('/descpredict')
async def predict(request: Request,MovieName: str = Form(...)):

    '''

    For rendering results on HTML GUI

    '''
    data = MovieName
    
    prediction_text = get_recommendations(data,indices,titles,cosine_sim)[:10]
    # print(prediction_text)
    return templates.TemplateResponse('Movie.html', context={'request': request, 'prediction_text': prediction_text})
# Meta Data Based Prediction
@app.post('/metadatapredict')
async def predict(request: Request,MovieName: str = Form(...)):

    '''

    For rendering results on HTML GUI

    '''
    data = MovieName
    # prediction_text = get_recommendations(data,indices,titles,cosine_sim).head(10)
    prediction_text = get_recommendations(data,indices_1,titles_1,cosine_sim_1)[:10]
    return templates.TemplateResponse('Movie.html', context={'request': request, 'prediction_text': prediction_text})


if __name__ == "__main__":

    app.run(debug=True)