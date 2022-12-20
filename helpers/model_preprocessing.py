"""_summary_

Returns:
    _type_: _description_
"""
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np

#pylint: disable=C0301


def prepare_sequences(notes: list=None, note_map: dict=None,
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


def prepare_note_duration_sequences(notes: list=None, note_map: dict=None, 
                                    durations: list=None, duration_map: dict=None,
                                    sequence_length: int=50, step: int=1):
    """_summary_

    Args:
        notes (list, optional): _description_. Defaults to None.
        note_map (dict, optional): _description_. Defaults to None.
        durations (list, optional): _description_. Defaults to None.
        duration_map (dict, optional): _description_. Defaults to None.
        sequence_length (int, optional): _description_. Defaults to 50.
        step (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    notes_features = []
    notes_targets = []
    notes_last_index = len(notes) - sequence_length
    duration_features = []
    duration_targets = []
    for i in range(0, notes_last_index, step):
        note_feature = notes[i:i + sequence_length]
        note_feature = [note_map[feat] for feat in note_feature]
        note_target = notes[i + sequence_length]
        note_target = note_map[note_target]
        duration_feature = durations[i:i + sequence_length]
        duration_feature = [duration_map[feat] for feat in duration_feature]
        duration_target = durations[i + sequence_length]
        duration_target = duration_map[duration_target]
        notes_features.append(note_feature)
        notes_targets.append(note_target)
        duration_features.append(duration_feature)
        duration_targets.append(duration_target)
    notes_features = np.array(notes_features)
    notes_targets = np.array(notes_targets)
    duration_features = np.array(duration_features)
    duration_targets = np.array(duration_targets)
    return notes_features, notes_targets, duration_features, duration_targets


def prepare_data_for_model(features: list=None,
                                    targets: list=None):
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
    x_orig = x_temp/x_temp.max()
    y_orig = to_categorical(targets)

    x_train, x_test, y_train, y_test = train_test_split(x_orig,  y_orig, test_size=0.2, random_state=101)

    return x_train, x_test, y_train, y_test


def prepare_notes_duration_for_model(notes_features, notes_targets, duration_features, duration_targets,
                                    length_notes, length_duration, test_size):
    """_summary_

    Args:
        notes_features (_type_): _description_
        notes_targets (_type_): _description_
        duration_features (_type_): _description_
        duration_targets (_type_): _description_
        length_notes (_type_): _description_
        length_duration (_type_): _description_

    Returns:
        _type_: _description_
    """
    dim1 = len(notes_targets)
    dim2 = len(notes_features[0])
    dim3 = len(duration_targets)
    dim4 = len(duration_features[0])
    notes_temp = np.reshape(notes_features, (dim1, dim2))
    duration_temp = np.reshape(duration_features, (dim3, dim4))
    notes_orig = notes_temp/length_notes
    duration_orig = duration_temp/length_duration
    notes_target_orig = to_categorical(notes_targets, num_classes=length_notes)
    duration_target_orig = to_categorical(duration_targets, num_classes=length_duration)
    assert notes_orig.shape[0] == duration_orig.shape[0]
    assert notes_target_orig.shape[0] == duration_target_orig.shape[0]
    model_inputs = [notes_orig[:-test_size], duration_orig[:-test_size]]
    model_outputs = [notes_target_orig[:-test_size], duration_target_orig[:-test_size]]
    test_inputs = [notes_orig[-test_size:], duration_orig[-test_size:]]
    test_outputs = [notes_target_orig[-test_size:], duration_target_orig[-test_size:]]


    return model_inputs, model_outputs, test_inputs, test_outputs
