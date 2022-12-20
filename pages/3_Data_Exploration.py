import streamlit as st
import pickle
import pandas as pd
from collections import Counter
from PIL import Image


st.header('Data Exploration')

with open('./data/pickled_midi/Alternative Rock', 'rb') as f:
    notes_with_duration = pickle.load(f)


notes_df = pd.DataFrame({'notes_with_duration': notes_with_duration})
st.write('List of notes in the music stream')
notes_df.iloc[:10, 0]

st.write('Frequency of notes used in the genre')
st.bar_chart(notes_df.groupby('notes_with_duration').value_counts().sort_values(ascending=False))

st.subheader('Pitch classes for a 50 Cent song')
pitch_img = Image.open('pitchclass_50cent.png')
st.image(pitch_img, caption='Histogram of Pitch Classes for song', width=500)

st.subheader('Pitch classes for the same song')
pitch_img2 = Image.open('pitchclass_offset.png')
st.image(pitch_img2, caption='Histogram of Pitch Classes for song', width=500)

st.subheader('Pitch classes for a generated sample from gangster rap')
pitch_img3 = Image.open('pitchclass.png')
st.image(pitch_img3, caption='Histogram of Pitch Classes for song', width=500)

st.subheader('Pitch classes offset for the gangster rap sample')
pitch_img4 = Image.open('predictedoffset.png')
st.image(pitch_img4, caption='Histogram of Pitch Classes for song', width=500)

st.subheader('Pitch classes offset for sample with rest')
pitch_img5 = Image.open('predictedoffsetrest.png')
st.image(pitch_img5, caption='Histogram of Pitch Classes for song', width=500)



st.text('Madonna Markov only whole notes')
audio2 = open('./data/generated_audio/markov.wav', 'rb')
audio2.read()
st.audio(audio2, format='audio/wav')

st.text('Gangster Rap Markov')
audio3 = open('./data/generated_audio/gangster_rap_markov.wav', 'rb')
audio3.read()
st.audio(audio3, format='audio/wav')

st.text('Dance Pop LSTM')
audio1 = open('./data/generated_audio/dancepop_latest.wav', 'rb')
audio1.read()
st.audio(audio1, format='audio/wav')

st.text('Reggae with rest LSTM')
audio1 = open('./data/generated_audio/reggae_with_rest.wav', 'rb')
audio1.read()
st.audio(audio1, format='audio/wav')