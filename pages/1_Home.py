import streamlit as st
from PIL import Image

st.header('Home')

spiced_image = Image.open('spicedlogo.png')

streamlit_image = Image.open('streamlit.png')

music21_image = Image.open('music21.jpeg')

tensorflow_image = Image.open('tensorflow.png')

sklearn_image = Image.open('sklearn.jpeg')

st.subheader('The main source of support through this project')
st.image(spiced_image, width=400)

st.subheader('The tech stack used in the project')

col1, col2 = st.columns(2)
col1.image(sklearn_image, width=200, caption='Python and other utility functions for data')

col2.image(streamlit_image, width=200, caption='Streamlit for web application')

col1.image(tensorflow_image, width=200, caption='Tensorflow and Keras for modeling and training')

col2.image(music21_image, width=200, caption='Music21 for extraction of data from midi files')





