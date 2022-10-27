import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from surprise import Reader, Dataset, SVD
import requests

st.title('MOVIE RECOMMENDATION 🎥')
st.write('HYBRID RECOMMENDATION 👻')
df= pd.read_csv(r"/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/data/final.csv")
hoo= load(open('/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/models/hoo.pkl', 'rb'))
svd= load(open('/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/models/svd.pkl', 'rb'))
tfidf_matrix= load(open('/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/models/tfidf_matrix.pkl', 'rb'))
cosine_sim= load(open('/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/models/cosine_sim.pkl', 'rb'))

indices = pd.Series(hoo.index, index=hoo['original_title']).drop_duplicates()


links_df = pd.read_csv('/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/data/links_small.csv')
col=np.array(links_df['tmdbId'], np.int64)
links_df['tmdbId']=col


links_df = links_df.merge(hoo[['title', 'tmdbId']], on='tmdbId').set_index('title')
links_index = links_df.set_index('tmdbId') 


def hybrid(userId, title):
    idx = indices[title]
    tmdbId = links_df.loc[title]['tmdbId']
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31] 
    movie_indices = [i[0] for i in sim_scores]
    
    movies = hoo.iloc[movie_indices][['title', 'vote_average', 'tmdbId']]
    movies['est'] = movies['tmdbId'].apply(lambda x: svd.predict(userId, links_index.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False) 
    movies.columns = ['Title', 'Vote Average', 'TMDb Id', 'Estimated Prediction']
    return movies.head(15)


userId=st.number_input('Enter userId')
title=st.selectbox("select a movie for recommendation",df["title"].values)

btn_click=st.button("RECOMMEND")
if btn_click == True:
    st.markdown("Showing Top movies according to most user ratings")
    tr=hybrid(userId, title)
    st.dataframe(tr)
