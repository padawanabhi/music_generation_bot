"""
    _summary_
"""
from helpers import data_loader as dl
from helpers import notes_helpers as nh
from helpers import model_preprocessing as mp
from helpers import generative_models as gm


# midi_paths, _ = dl.get_midi_file_list('./Reggae')


# midi_stream = dl.get_midi_stream(midi_paths=midi_paths)


# notes = nh.extract_notes(midi_stream=midi_stream)


# chord_count = nh.get_note_counts(notes=notes)


# rare_notes = nh.extract_rare_notes(chord_count, threshold=10)

# corpus = nh.remove_rare_notes(notes=notes, rate_notes=rare_notes)


# note_map, reverse_map, length = nh.create_feature_map(feature=notes)


# features, targets = mp.prepare_sequences(notes=notes, note_map=note_map, sequence_length=60, step=1)

# x_train, x_test, y_train, y_test = mp.prepare_data_for_model(features=features, targets=targets)


# chords_model = gm.GenerativeModel(x_train, x_test, y_train, y_test)

# chords_model.create_network()

# chords_model.compile_model()

# chords_model.print_summary()

# chords_model.train_model(model_name='normal_model',batch_size=250, epochs=100)

# music = chords_model.generate_new(sample_size=50, reverse_map=reverse_map, normalization_factor=length)

# melody = nh.extract_melody_from_notes(music)

# melody_stream = dl.create_midi_stream_from_notes(melody)

# dl.save_midi2file(melody_stream, file_name='normal_sample')

#dl.convert_midi2audio(midi_file_name='madonna_markov(good)', output_filename='madonna_markov')

