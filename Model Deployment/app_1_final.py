import streamlit as st
import pandas as pd
from pickle import load


data = pd.read_csv('MoviesData.csv')



C= data['vote_average'].mean()
m= data['vote_count'].quantile(0.9)
q_movies = data.loc[data['vote_count'] >= m]



def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
    
 
q_movies['rating_count'] = q_movies.apply(weighted_rating, axis=1)

def popularity_recommend():
    movie_df=q_movies[['title', 'vote_count', 'vote_average', 'rating_count']].reset_index(drop=True).head(10)
    return movie_df

st.title('MOVIE RECOMMENDATION')
st.write("POPULARITY BASED  RECOMMENDATION")

btn_click=st.button("CLICK FOR RECOMMENDATION")
if btn_click == True:
    st.markdown("Showing Top movies according to trend")
    tr=popularity_recommend()
    st.dataframe(tr)





