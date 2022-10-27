import streamlit as st
import pandas as pd
from pickle import load
similarity= load(open(r'/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/models/similarity_tf-002.pkl', 'rb'))

df= pd.read_csv(r"/home/chaitanyadubal/Downloads/FINL MOVIE RECOMMEND/Project/data/final.csv")

def recommend(movie):
    movies_list=[]
    index = df[df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        movies_list.append(df.iloc[i[0]].title)
    return movies_list



st.title('MOVIE RECOMMENDATION 🎥')
st.write("CONTENT BASED  RECOMMENDATION 📷✨")
option=st.selectbox("select a movie for recommendation",df["title"].values)
btn_click=st.button("RECOMMEND")
if btn_click == True:
    st.markdown("Showing Top movies according to movie choosed")
    tr=recommend(option)
    st.dataframe(tr)