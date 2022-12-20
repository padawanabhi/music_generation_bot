import streamlit as st
import time
import os
from run_model_with_duration import predict_new_samples
from markov_helpers import predict_new_markov



genres_with_rest = ['Disco', 'Glam Rock']
genres_without_rest = ['Alternative Rock', 'Electro', 'Glam Rock']

models = ['LSTM', 'Markov Chains']

st.header('Generate Music with AI')
model_name = st.selectbox('Choose Model', models)

include_rest = st.checkbox('Include Rest notes')

if not include_rest:
    genre = st.selectbox('Choose a Genre', genres_without_rest)
else:
    genre = genre = st.selectbox('Choose a Genre', genres_with_rest)

sample_size = st.number_input('Size of sample to create', min_value=0, step=5)

if st.button('Create Music'):
    if model_name == 'LSTM':
        if not include_rest:
            log = st.text('Running generation process')
            predict_new_samples(genre=genre, output_file='predicted_sample', 
                                    sample_size=int(sample_size), with_Rest=False)
            time.sleep(3)
            generated_audio = open('predicted_sample.wav', 'rb')
            generated_audio.read()
            log.text('New music generated')
            st.audio(generated_audio, format='audio/wav')
        else:
            log = st.text('Running generation process')
            predict_new_samples(genre=genre, output_file='predicted_rest_sample', 
                                sample_size=int(sample_size), with_Rest=True)
            time.sleep(3)
            generated_audio = open('predicted_rest_sample.wav', 'rb')
            generated_audio.read()
            log.text('New music generated')
            st.audio(generated_audio, format='audio/wav')
    elif model_name == 'Markov Chains':
        try:
            if not include_rest:
                log = st.text('Running generation process')
                predict_new_markov(genre=genre, sample_size=int(sample_size), with_Rest=False)
                time.sleep(3)
                generated_audio = open('predicted_markov.wav', 'rb')
                generated_audio.read()
                log.text('New music generated with Markov Chains')
                st.audio(generated_audio, format='audio/wav')
        except:
            log.text('Generation has been halted')
            st.write('Sorry! This one is not working currently')


