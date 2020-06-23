import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

from tmdbv3api import TMDb
tmdb = TMDb()
tmdb.api_key = '5492165c61b1a21c06eb3a3b578a6339'

from tmdbv3api import Movie

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_sim():
    data = pd.read_csv('main_data.csv')
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
def home():
    return render_template('home.html')


@app.route("/recommend")
def recommend():
    movie = request.args.get('movie') # get movie name from the URL
    r = rcmd(movie)
    movie = movie.upper()
    if type(r)==type('string'): # no such movie found in the database
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        tmdb_movie = Movie()
        result = tmdb_movie.search(movie)
        movie_id = result[0].id
        movie_name = result[0].title
        reviews_list = []
        reviews_status = []
        response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
        data_json = response.json()
        imdb_id = data_json['imdb_id']
        poster = data_json['poster_path']
        img_path = 'https://image.tmdb.org/t/p/original{}'.format(poster)
        if data_json['genres']:
            genres = []
            genre_str = ", " 
            for i in range(0,len(data_json['genres'])):
                genres.append(data_json['genres'][i]['name'])
            genre = genre_str.join(genres)
        sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
        soup = bs.BeautifulSoup(sauce,'lxml')
        soup_result = soup.find_all("div",{"class":"text show-more__control"})
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = clf.predict(movie_vector)
                reviews_status.append('Good' if pred else 'Bad')
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))} 
        vote_count = "{:,}".format(result[0].vote_count)
        # convert 10-06-2019 to June 10 2019
        MONTHS = ['January', 'February', 'Match', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
        def date_convert(s):
            y = s[:4]
            m = int(s[5:-3])
            d = s[8:]
            month_name = MONTHS[m-1]

            result= month_name + ' ' + d + ' '  + y
            return result
        rd = date_convert(result[0].release_date)
        status = data_json['status']
        # convert minutes to hours minutes (eg. 148 minutes to 2 hours 28 mins)
        if data_json['runtime']%60==0:
            runtime = "{:.0f} hours".format(data_json['runtime']/60)
        else:
            runtime = "{:.0f} hours {} minutes".format(data_json['runtime']/60,data_json['runtime']%60)
        poster = []
        movie_title_list = []
        # getting the posters for the recommended movies
        for movie_title in r:
            list_result = tmdb_movie.search(movie_title)
            movie_id = list_result[0].id
            response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
            data_json = response.json()
            poster.append('https://image.tmdb.org/t/p/original{}'.format(data_json['poster_path']))
        movie_cards = {poster[i]: r[i] for i in range(len(r))}
        return render_template('recommend.html',movie=movie,mtitle=r,t='l',cards=movie_cards,
            result=result[0],reviews=movie_reviews,img_path=img_path,genres=genre,vote_count=vote_count,
            release_date=rd,status=status,runtime=runtime)

if __name__ == '__main__':
    app.run(debug=True)
