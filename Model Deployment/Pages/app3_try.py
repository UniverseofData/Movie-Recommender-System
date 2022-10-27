from logging import PlaceHolder
from tkinter import CENTER
import streamlit as st
import pandas as pd
from pickle import load
import requests

st.title('MOVIE RECOMMENDATION 🎥')
st.write("COLLABRATIVE FILTERING RECOMMENDATION")


movie_df = pd.read_csv("/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/data/movies_metadata.csv")
links_df = pd.read_csv("/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/data/links.csv")
movie_df['imdb_id'] = movie_df['imdb_id'].apply(lambda x: str(x)[2:].lstrip("0"))
links_df['imdbId'] = links_df['imdbId'].astype(str)


KNN = load(open('/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/models/algo_KNN.pkl', 'rb'))


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
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in top_recommended_movies.title:
        recommended_movie_names.append(i)

    for i in top_recommended_movies.imdb_id:
        recommended_movie_posters.append(fetch_poster(i))

    return recommended_movie_names,recommended_movie_posters


def fetch_poster(movie_id):
    response=requests.get('https://api.themoviedb.org/3/movie/tt0{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data=response.json()
    print(data)
    if "poster_path"in data:
        full_path = "https://image.tmdb.org/t/p/w500"+ data["poster_path"]
        return full_path
    
    

# KNN predictions
pred_KNN = prediction(KNN)
# recommended_movies_KNN, top_recommended_movies_KNN = top_recommendations(pred_KNN, 3)

x=st.number_input('Enter Number of User Rated')
btn_click=st.button("RECOMMEND")
if btn_click == True:
    st.markdown("Showing Top movies according to most user ratings")
    names,posters=top_recommendations(pred_KNN,x)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(names[0])
        st.image(posters[0])
    with col2:
        st.text(names[1])
        st.image(posters[1])

    with col3:
        st.text(names[2])
        st.image(posters[2])
    with col4:
        st.text(names[3])
        st.image(posters[3])
    with col5:
        st.text(names[4])
        st.image(posters[4])


    

    
    


st.write("*"*50)


