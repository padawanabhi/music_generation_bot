"""_summary_
"""
import os
import pickle
import glob
from music21 import converter, stream
from midi2audio import FluidSynth
from helpers.notes_helpers import extract_notes_and_durations

MIDI_BASE_FOLDER = '/Users/abhi/Desktop/spiced_projects/final_project/data/midi_with_genres'

PROJECT_BASE_PATH = '/Users/abhi/Desktop/spiced_projects/final_project'

STORE_FOLDER = '/Users/abhi/Desktop/spiced_projects/final_project/data/pickled_midi/'

def get_midi_file_list(folder_name: str) -> tuple:
    """_summary_

    Args:
        folder_name (str): Name of folder from where to extract midi file paths. 
                            Folder name inside ./data/midi_with_genres/

    Returns:
        list : A list of midi file paths
        list: List of targets (genres) for each midi file
    """
    # folder, subfolder = folder_name.split('/')
    # folder_path = MIDI_BASE_FOLDER + folder_name
    # os.chdir(folder_path)
    # for file in os.listdir():
    #     if file.endswith('.mid'):
    #         print(f'{folder_path}/{file}')
    # os.chdir(MIDI_BASE_FOLDER+folder_name)
    midi_path_list = []
    targets = []
    if '/' in folder_name:
        sub_genre = folder_name.split('/')[1]
    else:
        sub_genre = folder_name
    # for root, _, files in os.walk(f'{folder_name}/', topdown=True):
    #     for name in files:
    #         if name != '.DS_Store':
    #             file_path = os.path.join(MIDI_BASE_FOLDER, root, name)
    #             print(file_path)
    for testfile in glob.glob(f'{MIDI_BASE_FOLDER}/{folder_name}/**/*.mid', recursive=True):
        midi_path_list.append(testfile)
        targets.append(sub_genre)
    # os.chdir(PROJECT_BASE_PATH)
    
    return midi_path_list, targets


def get_midi_stream(midi_paths: list) -> list:
    """_summary_

    Args:
        midi_paths (list): List of midi file paths to create music21 streams 

    Returns:
        list: A list of music21.stream.Stream objects
    """
    midi_streams = []

    for file in midi_paths:
        try:
            song_midi = converter.parse(file)
            midi_streams.append(song_midi)
        except:
            continue
    return midi_streams


def get_chordified_stream(midi_paths: list) -> list:
    """_summary_

    Args:
        midi_paths (list): List of midi file paths to create music21 streams 

    Returns:
        list: A list of choridified music21.stream.Stream objects
    """
    midi_streams = []

    for file in midi_paths:
        song_midi = converter.parse(file).chordify()
        midi_streams.append(song_midi)

    return midi_streams


def create_midi_stream_from_notes(melody: list) -> stream.Stream:
    """_summary_

    Args:
        melody (list): _description_

    Returns:
        stream.Stream: _description_
    """
    
    melody_stream = stream.Stream(melody)

    return melody_stream


def save_midi2file(melody_stream : stream.Stream,  file_name: str='generated_music') -> None:
    """_summary_

    Args:
        melody_stream (stream.Stream): _description_
        file_name (str, optional): _description_. Defaults to 'generated_music'.
    """

    melody_stream.write('midi', f'./data/generated_midi/{file_name}.mid')


def convert_midi2audio_old(midi_file_name: str, output_filename= str) -> None:
    """_summary_

    Args:
        midi_file_path (str): _description_
        output_filename (_type_, optional): _description_. Defaults to str.
    """

    fs = FluidSynth()
    fs.midi_to_audio(f'./data/generated_midi/{midi_file_name}.mid', 
                        f'./data/generated_audio/{output_filename}.wav')


def convert_midi2audio(midi_file_name: str, output_filename= str) -> None:
    """_summary_

    Args:
        midi_file_path (str): _description_
        output_filename (_type_, optional): _description_. Defaults to str.
    """

    fs = FluidSynth()
    fs.midi_to_audio(f'{midi_file_name}.mid', 
                        f'{output_filename}.wav')



if __name__ == '__main__':

    print(os.getcwd())
    X, y = get_midi_file_list('Rap/Gangster Rap')

    print(X)
    print(y)
    print(os.getcwd())