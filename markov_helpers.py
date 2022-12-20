"""_summary_
"""
from music21 import note, chord, stream
import numpy as np
import pandas as pd
from run_model_with_duration import load_note_duration
from helpers.notes_helpers import create_feature_map
from helpers.data_loader import create_midi_stream_from_notes, save_midi2file, convert_midi2audio_old

STORE_FOLDER = '/Users/abhi/Desktop/spiced_projects/final_project/data/pickled_with_rest/'

STORE_FOLDER_NORMAL = '/Users/abhi/Desktop/spiced_projects/final_project/data/pickled_midi/'

np.random.seed(101)

def extract_note_duration_markov(midi_stream: list=None) -> list:
    for data in midi_stream:
        notes = []
        duration = []
        for element in data.flat:
            if isinstance(element, note.Note):
                notes.append(str(element.nameWithOctave))
                duration.append(element.duration.quarterLength)
            if isinstance(element, note.Rest):
                notes.append(element.name)
                duration.append(element.duration.quarterLength)
            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                duration.append(element.duration.quarterLength)

    notes_with_duration = []
    for zip_inp in zip(notes, duration):
       notes_with_duration.append(zip_inp)

    return notes_with_duration


def get_next_feature(notes_with_duration, feature_map):
    encoded_notes = [feature_map[char] for char in notes_with_duration]
    song_data = pd.DataFrame({'encoded_note': encoded_notes})
    song_data['next_encoded_note'] = song_data['encoded_note'].shift(-1)
    song_data['next_encoded_note'].fillna(method='ffill', inplace=True)
    
    return encoded_notes, song_data


def get_transition_matrix(dataframe):
    notes_matrix = pd.crosstab(dataframe['encoded_note'], 
                                dataframe['next_encoded_note'], normalize=1)

    return notes_matrix


def generate_music_markov(input: list=None, notes_matrix: pd.DataFrame=None, reverse_map: dict=None,sequence_length: int=30):
    first_note = np.random.choice(input)
    generated_notes = []
    generated_notes.append(reverse_map[first_note])
    for _ in range(sequence_length):
        next_encoded_note = np.random.choice(a=np.array(notes_matrix.columns), 
                                    p=notes_matrix.iloc[first_note])
        next_note = reverse_map[next_encoded_note]
        generated_notes.append(next_note)

    return generated_notes


def create_melody_markov(notes_list):
    melody = []
    for (element, duration) in notes_list:
        if '.' in element:
            notes = element.split('.')
            temp = chord.Chord(notes)
            temp.quarterLength = duration
            melody.append(temp)
        elif element != 'rest':
            temp = note.Note(element)
            temp.quarterLength = duration
            melody.append(temp)
        else:
            temp = note.Rest()
            temp.quarterLength = duration
            melody.append(temp)

    return melody


def create_midi_from_melody(melody: list, file_name: str) -> None:
    """_summary_

    Args:
        melody (list): _description_
        file_name (str): _description_
    """
    melody_stream = stream.Stream(melody)

    melody_stream.write('midi', f'./data/generated_midi/{file_name}')


def predict_new_markov(genre: str, sample_size: int, with_Rest: bool):
    file_name = ''.join(genre.lower().split()) + '_markov'
    if not with_Rest:
        notes = load_note_duration(genre=genre, with_Rest=False)
    else:
        notes = load_note_duration(genre=genre, with_Rest=True)
    feat_map, feat_rev_map, feature_length = create_feature_map(feature=notes)
    encoded_notes, notes_df = get_next_feature(notes, feat_map)
    trans_matrix = get_transition_matrix(notes_df)
    music = generate_music_markov(encoded_notes, trans_matrix, feat_rev_map, sequence_length=sample_size)
    melody = create_melody_markov(music)
    melody_stream = create_midi_stream_from_notes(melody=melody)
    save_midi2file(melody_stream=melody_stream, file_name=file_name)
    convert_midi2audio_old(file_name, file_name)

if __name__ == '__main__':

    genre = 'Disco'
    file_name = ''.join(genre.lower().split()) + '_markov'
    notes = load_note_duration(genre=genre, with_Rest=False)
    feat_map, feat_rev_map, feature_length = create_feature_map(feature=notes)
    encoded_notes, notes_df = get_next_feature(notes, feat_map)
    trans_matrix = get_transition_matrix(notes_df)
    music = generate_music_markov(encoded_notes, trans_matrix, feat_rev_map, sequence_length=100)
    melody = create_melody_markov(music)
    melody_stream = create_midi_stream_from_notes(melody=melody)
    save_midi2file(melody_stream=melody_stream, file_name=file_name)
    convert_midi2audio_old(file_name, file_name)
