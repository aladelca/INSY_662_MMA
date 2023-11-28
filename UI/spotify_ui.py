import time
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time

DATA = 'clean_songs_en_fr_sp.xlsx'


@st.cache_data
def load_data(nrows):
    data = pd.read_excel(DATA, nrows=nrows)  # Use pd.read_excel to read Excel files
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data = load_data(10000)

#side_bar
with st.sidebar:
        st.image('logo.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        st.subheader(" ")
        st.subheader("Home")
        st.subheader("Search")
        st.subheader("Your Library")



#image = st.image('logo.jpg', caption='Your Image Caption', use_column_width=True)

st.title('How are you feeling today?')
#Get user input 
user_input = st.text_input("Describe your mood in less than 10 words", value = " ")
st.subheader(" ")
search_button = st.button("Recommend songs")
st.subheader(" ")


if search_button:
    data_load_state = st.text('Analyzing your mood...')
    time.sleep(3)
    data_load_state.empty()
    st.markdown("Your polarity score is: -0.35")   #Input polarity score here
    time.sleep(2)
    st.markdown("Your mood is: Melancholic")             #Input mood here
    st.subheader(" ")
    time.sleep(2)
    data_load_state = st.text('Loading songs...')
    time.sleep(5)
# Update the text after the sleep
    data_load_state.markdown('<span style="color:#1DB954">Songs loaded!</span>', unsafe_allow_html=True)
    time.sleep(2)
    data_load_state.empty()
    


    st.subheader("Here's your songs list:")

    data_df = pd.DataFrame(
    {

        "apps": [
            "https://upload.wikimedia.org/wikipedia/en/thumb/c/cb/PinkFloydAnotherBrickCover.jpg/220px-PinkFloydAnotherBrickCover.jpg",
            "https://i.ytimg.com/vi/cbRBORpHv4o/maxresdefault.jpg",
            "https://i.ytimg.com/vi/_OKbYZ2qFrQ/maxresdefault.jpg",
            "https://i1.sndcdn.com/artworks-000086063252-05dnz6-t500x500.jpg",
            "https://i1.sndcdn.com/artworks-xc01bDEoFyo7LEFO-1sONQw-t500x500.jpg",
        ],
        
        "Title": [
            "Another Brick in the Wall, Pt. 2",
            "Sick Feeling",
            "Psycho",
            "Travesuras - Remix",
            "Chulo pt.2"
        ],
        
        "Artist":["Pink Floyd", "boy pablo", "Gavin Magnus", "Nio Garcia","Bad Gyal"],
                  
        "Polarity":["-0.39",'-0.38','-0.39','-0.45','-0.35']
    }
)

    st.data_editor(
        data_df,
        column_config={
            "apps": st.column_config.ImageColumn(
                "Songs", width='medium', help="Preview"
            )
        },
        hide_index=True,
    )


#Show all raw data
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
