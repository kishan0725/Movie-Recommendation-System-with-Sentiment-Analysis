import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle

from tmdbv3api import TMDb
tmdb = TMDb()
tmdb.api_key = '5492165c61b1a21c06eb3a3b578a6339'

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

from tmdbv3api import Movie
tmdb_movie = Movie()

def create_sim():
    data = pd.read_csv('final_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data,sim


def rcmd(m):
    m = m.lower()
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        data, sim = create_sim()
    # check if the movie is in our database or not
    if m not in data['movie_title'].unique():
        return('Sorry! This movie is not in our database. Please check the spelling or try with some other movies')
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title']==m].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

app = Flask(__name__)

@app.route("/")
def login():
    return render_template('login.html')

@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')

@app.route("/movie/<movie_title>")
def movie(movie_title):
    reviews_list = []
    reviews_status = []
    lower = movie_title.lower()
    movie_name = lower.replace(" ","_").replace(":","").replace("-","").replace("'","")
    sauce = urllib.request.urlopen('https://www.rottentomatoes.com/m/{}/reviews'.format(movie_name)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"the_review"})
    for reviews in soup_result:
        if reviews.string.replace("\n",""):
            reviews_list.append(reviews.string.replace("\n",""))
            movie_review_list = np.array([reviews.string.replace("\n","")])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))} 
    result = tmdb_movie.search(movie_title)
    return render_template('movie.html',scrape_movie=movie_name,result=result[0],reviews=movie_reviews)

if __name__ == '__main__':
    app.run(debug=True)
