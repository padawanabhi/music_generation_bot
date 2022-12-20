"""_summary_

Returns:
    _type_: _description_
"""
import os
import pickle
from collections import Counter
from music21 import instrument, note, chord

STORE_FOLDER = '/Users/abhi/Desktop/spiced_projects/final_project/data/pickled_midi/'


def extract_notes(midi_stream: list, with_Rest=False) -> list:
    """_summary_

    Args:
        midi_stream (list): A list of music21.stream.Stream objects
        with_rest (bool, optional): To include note.Rest class. Defaults to False.

    Returns:
        list: A list of note/chord/rest strings
    """
    notes = []
    pick = None

    for song in midi_stream:
        songs = instrument.partitionByInstrument(song)
        for part in songs.parts:
            pick = part.recurse()
            for element in pick:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                if isinstance(element, chord.Chord):
                    notes.append(".".join(n.nameWithOctave for n in element.pitches))
                if with_Rest and isinstance(element, note.Rest):
                    notes.append(element.name)
                    
    return notes


def extract_melody_from_notes(notes: list) -> list:
    """_summary_

    Args:
        notes (list): A list of note/chord/rest strings

    Returns:
        list: List of music21 objects
    """
    melody = []
    offset = 0
    for element in notes:
        if ("." in element or element.isdigit()):
            chord_notes = element.split(".")
            new_notes = []
            for temp in chord_notes:
                note_snip = note.Note(temp)
                new_notes.append(note_snip)
                chord_snip = chord.Chord(new_notes)
                chord_snip.offset = offset
                melody.append(chord_snip)
        elif element != 'rest':
            note_snip = note.Note(element)
            note_snip.offset = offset
            melody.append(note_snip)
        else:
            note_snip = note.Rest()
            note_snip.offset = offset
            melody.append(note_snip)
        offset += 1

    return melody


def get_note_counts(notes: list) -> Counter:
    """_summary_

    Args:
        notes (list): List of note/chord/rest strings

    Returns:
        Counter: Dictionary of unique notes as keys with total count in list as values
    """
    note_counts = Counter(notes)

    return note_counts


def extract_rare_notes(note_count_dict: dict, threshold: int=10) -> list:
    """_summary_

    Args:
        count_dict (dict, optional): _description_. Defaults to None.
        threshold (int, optional): Threshold value for note count, below 
                                    which note will be considered rare. Defaults to 10.

    Returns:
        list: List of notes with values less than threshold
    """
    rare_notes = []
    for key, value in note_count_dict.items():
        if value < threshold:
            rare = key
            rare_notes.append(rare)

    return rare_notes


def remove_rare_notes(notes: list, rate_notes: list) -> list:
    """_summary_

    Args:
        notes(list): List of all notes from the stream.
        rate_notes (list): List of rare notes from the stream.

    Returns:
        list: List of notes excluding the rare notes.
    """
    for element in notes:
        if element in rate_notes:
            notes.remove(element)

    return notes



def extract_notes_and_durations(midi_stream: list, with_Rest=False) -> tuple:
    """_summary_

    Args:
        midi_stream (list): List of music21.stream objects

    Returns:
        tuple (list, list): List of notes and list of durations 
    """
    notes = []
    duration = []
    for data in midi_stream:
        for element in data.flat:
            if isinstance(element, note.Note):
                notes.append(str(element.nameWithOctave))
                duration.append(element.duration.quarterLength)
            if isinstance(element, chord.Chord):
                notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                duration.append(element.duration.quarterLength)
            if with_Rest and isinstance(element, note.Rest):
                    notes.append(element.name)
                    duration.append(element.duration.quarterLength)
    
    return (notes, duration)


def create_melody_from_notes_durations(notes: list, durations: list) -> list:
    """_summary_

    Args:
        notes (list): _description_
        durations (list): _description_

    Returns:
        list: _description_
    """
    melody = []
    for element, quater_length in zip(notes, durations):
        if ("." in element or element.isdigit()):
            chord_notes = element.split(".")
            new_notes = []
            for temp in chord_notes:
                note_snip = note.Note(temp)
                new_notes.append(note_snip)
                chord_snip = chord.Chord(new_notes)
                chord_snip.quarterLength = quater_length
                melody.append(chord_snip)
        elif element != 'rest':
            note_snip = note.Note(element)
            note_snip.quarterLength = quater_length
            melody.append(note_snip)
        else:
            note_snip = note.Rest()
            note_snip.quarterLength = quater_length
            melody.append(note_snip)

    return melody


def create_feature_map(feature):
    """_summary_

    Args:
        feature (_type_): _description_

    Returns:
        _type_: _description_
    """
    unique_features = list(set(feature))

    feature_length = len(unique_features)

    feat_map = dict({v: k for k,v in enumerate(unique_features)})
    feat_rev_map = dict({k: v for k,v in enumerate(unique_features)})

    return feat_map, feat_rev_map, feature_length



if __name__ == '__main__':
    ...