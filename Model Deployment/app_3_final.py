from logging import PlaceHolder
from tkinter import CENTER
import streamlit as st
import pandas as pd
from pickle import load
import requests

st.title('MOVIE RECOMMENDATION')
st.write("COLLABRATIVE FILTERING RECOMMENDATION")


movie_df = pd.read_csv("movies_metadata.csv")
links_df = pd.read_csv("links.csv")
movie_df['imdb_id'] = movie_df['imdb_id'].apply(lambda x: str(x)[2:].lstrip("0"))
links_df['imdbId'] = links_df['imdbId'].astype(str)


KNN = load(open('models/algo_KNN.pkl', 'rb'))


def prediction(algo, users_K=10):
    pred_list = []
    for userId in range(1,users_K):
        for movieId in range(1,9067):
            rating = algo.predict(userId, movieId).est
            pred_list.append([userId, movieId, rating])
    pred_df = pd.DataFrame(pred_list, columns = ['userId', 'movieId', 'rating'])
    return pred_df

def top_recommendations(pred_df, top_N):
    link_movie = pd.merge(pred_df, links_df, how='inner', left_on='movieId', right_on='movieId')
    recommended_movie = pd.merge(link_movie, movie_df, how='left', left_on='imdbId', right_on='imdb_id')[['userId', 'movieId', 'rating','imdb_id','title']]
    sorted_df = recommended_movie.groupby(('userId'), as_index = False).apply(lambda x: x.sort_values(['rating'], ascending = False)).reset_index(drop=True)
    top_recommended_movies = sorted_df.groupby('userId').head(top_N)
    return top_recommended_movies


# KNN predictions
pred_KNN = prediction(KNN)
# recommended_movies_KNN, top_recommended_movies_KNN = top_recommendations(pred_KNN, 3)

x=st.number_input('Enter Number of User Rated')
btn_click=st.button("RECOMMEND")
if btn_click == True:
    st.markdown("Showing Top movies according to most user ratings")
    tr=top_recommendations(pred_KNN,x)
    columnD = tr.loc[:,'title']
    st.dataframe(tr)
    
    


st.write("*"*50)


