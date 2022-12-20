"""
    _summary_
"""
import pickle
import os
import glob
import numpy as np
from music21 import note, chord, stream, instrument, converter
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.utils.np_utils import to_categorical
from helpers.data_loader import get_midi_file_list, convert_midi2audio, convert_midi2audio_old
from helpers.notes_helpers import create_feature_map
from helpers.generative_models import GenerativeModel


STORE_FOLDER = '/Users/abhi/Desktop/spiced_projects/final_project/data/pickled_with_rest/'

STORE_FOLDER_NORMAL = '/Users/abhi/Desktop/spiced_projects/final_project/data/pickled_midi/'

def get_midi_stream_with_parts(midi_paths: list=None) -> list:
    """_summary_

    Args:
        midi_paths (list, optional): _description_. Defaults to None.

    Returns:
        list: _description_
    """
    midi_streams = []

    for file in midi_paths:
        song_midi = converter.parse(file)
        print(song_midi)
        midi_streams.append(song_midi)

    print(len(midi_streams))
    return midi_streams

def get_midi_streams(midi_paths: list, genre: str) -> list:
    """_summary_

    Args:
        midi_paths (list, optional): _description_. Defaults to None.

    Returns:
        list: _description_
    """
    notes = []
    for file in midi_paths:
        try:
            song_midi = converter.parse(file).chordify()
            notes.append(song_midi)
        except KeyboardInterrupt:
            break
        except Exception:
            print(f'{file} could not be parsed')
        
    return notes

def extract_note_duration(midi_stream: list, genre: str, with_Rest: bool=False) -> list:
    notes = []
    duration = []
    notes_with_duration = []
    if '/' in genre:
        sub_genre = genre.split('/')[1]
    else:
        sub_genre = genre
    for stream_obj in midi_stream:
        for element in stream_obj.flat:
            if isinstance(element, note.Note):
                notes.append(str(element.nameWithOctave))
                duration.append(element.duration.quarterLength)
            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                duration.append(element.duration.quarterLength)
            if with_Rest and isinstance(element, note.Rest):
                notes.append(element.name)
                duration.append(element.duration.quarterLength)

    for zip_inp in zip(notes, duration):
        notes_with_duration.append(zip_inp)

    if os.path.exists(os.path.join(STORE_FOLDER, sub_genre)):
        with open(os.path.join(STORE_FOLDER, sub_genre), 'ab') as f:
            pickle.dump(notes_with_duration, f)
    else:
        with open(os.path.join(STORE_FOLDER, sub_genre), 'xb') as f:
            pickle.dump(notes_with_duration, f)
         
    
    return notes_with_duration


def load_note_duration(genre: str, with_Rest: bool):
    if '/' in genre:
        sub_genre = genre.split('/')[1]
    else:
        sub_genre = genre
    if not with_Rest:
        with open(os.path.join(STORE_FOLDER_NORMAL, sub_genre), 'rb') as f:
                notes_with_duration = pickle.load(f)
    else:
        with open(os.path.join(STORE_FOLDER, sub_genre), 'rb') as f:
                notes_with_duration = pickle.load(f)
    
    return notes_with_duration


def extract_notes_by_part(midi_stream: list= None, with_Rest: bool=False) -> list:
    """_summary_

    Args:
        midi_stream (list, optional): _description_. Defaults to None.

    Returns:
        list: _description_
    """
    notes = []
    duration = []
    inst_part = []
    pick = None

    for song in midi_stream:
        print(song)
        songs = instrument.partitionByInstrument(song)
        print(songs)
        for part in songs.parts:
            print(part)
            print(part.id)
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.nameWithOctave))
                    duration.append(element.duration.quarterLength)
                    inst_part.append(part.id)
                if isinstance(element, chord.Chord):
                    notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                    duration.append(element.duration.quarterLength)
                    inst_part.append(part.id)
                if with_Rest and isinstance(element, note.Rest):
                    notes.append(element.name)
                    duration.append(element.duration.quarterLength)
                    inst_part.append(part.id)

    all_parts = list(set(inst_part))
    notes_with_duration = []
    for zip_inp in zip(notes, duration, inst_part):
       notes_with_duration.append(zip_inp)

    print(notes_with_duration[:5])
    return notes_with_duration, all_parts


def extract_notes(midi_stream: list=None) -> list:
    for data in midi_stream:
        notes = []
        duration = []
        for element in data.flat:
            if isinstance(element, note.Note):
                notes.append(str(element.nameWithOctave))
                duration.append(element.duration.quarterLength)
            # if isinstance(element, note.Rest):
            #     notes.append(element.name)
            #     duration.append(element.duration.quarterLength)
            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                duration.append(element.duration.quarterLength)

    notes_with_duration = []
    for zip_inp in zip(notes, duration):
       notes_with_duration.append(zip_inp)

    return notes_with_duration

def create_feature_map(notes_with_duration):
    note_set = list(set(notes_with_duration))

    notes_length = len(note_set)

    feat_map = dict({v: k for k,v in enumerate(note_set)})
    feat_rev_map = dict({k: v for k,v in enumerate(note_set)})

    return feat_map, feat_rev_map, notes_length


def prepare_encoded_sequences(notes: list=None, note_map: dict=None,
                        sequence_length: int=50, step: int=1):
    """_summary_

    Args:
        list (_type_): _description_
        notes (_type_, optional): _description_. Defaults to None,
                note_map: dict=None, sequence_length: int=50, step: int=1)->tuple(list.

    Returns:
        _type_: _description_
    """
    features = []
    targets = []
    last_index = len(notes) - sequence_length
    for i in range(0, last_index, step):
        feature = notes[i:i + sequence_length]
        feature = [note_map[feat] for feat in feature]
        target = notes[i + sequence_length]
        target = note_map[target]

        features.append(feature)
        targets.append(target)
    features = np.array(features)
    targets = np.array(targets)
    return features, targets


def prepare_encoded_data_for_model(features: list=None,
                                    targets: list=None, test_size=0.2, noramlization_factor: int=None):
    """_summary_

    Args:
        features (list, optional): _description_. Defaults to None.
        targets (list, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    dim1 = len(targets)
    dim2 = len(features[0])
    x_temp = np.reshape(features, (dim1, dim2, 1))
    x_orig = x_temp/noramlization_factor
    y_orig = to_categorical(targets)

    x_train, x_test, y_train, y_test = train_test_split(x_orig,  y_orig, test_size=test_size, shuffle=True, random_state=101)

    return x_train, x_test, y_train, y_test



def create_melody_stream(notes_list):
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
        
    melody_stream = stream.Stream(melody)
    print(len(melody))
    return melody, melody_stream


def create_melody_with_parts_stream(notes_list, part_list: list):
    # melody = []
    melody_parts = []
    for temp_part in part_list:
        melody_parts.append(stream.Part(id=temp_part))
    print(melody_parts)
    for temp_part in melody_parts:
        for (element, duration, parts) in notes_list:
            if temp_part.id == parts:
                if '.' in element:
                    notes = element.split('.')
                    temp = chord.Chord(notes)
                    temp.quarterLength = duration
                    temp_part.append(temp)
                    # melody.append(parts)
                elif element != 'rest':
                    temp = note.Note(element)
                    temp.quarterLength = duration
                    temp_part.append(temp)
                    # melody.append(parts)
                else:
                    temp = note.Rest()
                    temp.quarterLength = duration
                    temp_part.append(temp)
                    # melody.append(parts)
        
    melody_stream = stream.Stream(melody_parts)
    print(len(melody_parts))
    return melody_parts, melody_stream

def save_pickled_notes_durations(genres: list[str]) -> None:

    for genre in genres:
        midi_paths, labels = get_midi_file_list(genre)

        midi_streams = get_midi_streams(midi_paths, genre)

        notes = extract_note_duration(midi_stream=midi_streams, genre=genre, with_Rest=True)

        loaded_notes = load_note_duration(genre=genre)

        print(loaded_notes[:5])
        assert notes == loaded_notes


def predict_new_samples(genre: str, output_file: str, sample_size: int, with_Rest: bool):

    file_name = ''.join(genre.lower().split())

    print(file_name)

    notes = load_note_duration(genre, with_Rest=with_Rest)

    note_map, note_reverse_map, notes_length = create_feature_map(notes)

    features, targets = prepare_encoded_sequences(notes, note_map, 40, 1)

    x_train, x_test, y_train, y_test = prepare_encoded_data_for_model(features, targets, 0.2, notes_length)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    
    model = GenerativeModel(x_train, x_test, y_train, y_test)

    model.create_network()
    model.compile_model()

    if not with_Rest:
        model.model.load_weights(f'./model/{file_name}_model.h5')
    else:
        model.model.load_weights(f'./model/{file_name}_rest_model.h5')

    music = model.generate_new(sample_size=sample_size, reverse_map=note_reverse_map, normalization_factor=notes_length)

    melody, melody_stream = create_melody_stream(notes_list=music)

    melody_stream.write('midi', f'{output_file}.mid')

    convert_midi2audio(output_file, output_file)




if __name__ == '__main__':

    genre = 'Glam Rock'

   # save_pickled_notes_durations(genres=genres)

    file_name = ''.join(genre.lower().split())

    print(file_name)

    notes = load_note_duration(genre)

    note_map, note_reverse_map, notes_length = create_feature_map(notes)

    features, targets = prepare_encoded_sequences(notes, note_map, 40, 1)

    x_train, x_test, y_train, y_test = prepare_encoded_data_for_model(features, targets, 0.2, notes_length)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    
    model = GenerativeModel(x_train, x_test, y_train, y_test)

    K.clear_session()

    model.create_network()
    model.compile_model()

    #model.model.load_weights('./model/electro_model.h5')
    

    print(model.print_summary())

    model.train_model(model_name=f'{file_name}_rest_model',batch_size=120, epochs=200, patience=50)

    music = model.generate_new(sample_size=250, reverse_map=note_reverse_map, normalization_factor=notes_length)

    melody, melody_stream = create_melody_stream(notes_list=music)

    melody_stream.write('midi', f'./data/generated_midi/{file_name}_rest_sample.mid')

    #convert_midi2audio(f'{file_name}_sample', f'{file_name}_sample')








