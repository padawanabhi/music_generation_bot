import streamlit as st
from PIL import Image



st.header('Introduction')

st.subheader('What is Music?')

st.markdown("""
Music is an art form that combines either vocal or instrumental 
sounds, sometimes both, using form, harmony, and expression of emotion 
to convey an idea.  \n
**Characteristics of Music** \n
Music is made up of many components. These components could be classified as characteristics
of the concept itself. The characteristics of music can be explained by organizing them into 
categories.

1. Sound (timbre, pitch, duration, amplitude, overtone)
2. Melody
3. Harmony
4. Rhythm
5. Texture
6. Expression (dynamics, tempo, articulation) \n \n \n
""")

st.subheader('Representation of Music as notations')
img = Image.open('sheet_music.png')
st.image(img, caption='sheet music', width=400)

st.subheader('Structure of an actual song')
img2 = Image.open('midi_with_tracks.png')
st.image(img2, caption='sheet music', width=600)
st.text("Songs can have multiple instruments and multiple tracks in one file")

st.subheader('Structure of a song generated')
img3 = Image.open('Markov_midi.png')
st.image(img3, caption='sheet music', width=600)
st.text('Currently what we will generate is only one track with only piano')

st.subheader("""How can we generate music with machine learning ?
            """)

st.text(""" Music is essentially a sequence of notes/chords. \n
We take the smaller squences of these notes and chords and try to 
predict the next note or chord. \n

These sequence of notes and chords are then encoded into numbers before we can make predictions.
""")

img = Image.open('encoder.png')
st.image(img, width=500, caption='The encoding and decoding of music data (source: google images)')


st.write('A LSTM unit')
img3 = Image.open('lstm.png')
st.image(img3, caption='LSTM RNN Unit (source: towardsdatascience.com)', width=500)

st.text("""\n
After we have the encoded sequences, we provide the data to our model and train 
the model to predict the next number in the sequence
""")

img2 = Image.open('number-sequence.jpeg')
st.image(img2, caption='Predict the next number', width=500)

