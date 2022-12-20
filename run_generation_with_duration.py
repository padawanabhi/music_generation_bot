
from helpers import data_loader as dl
from helpers import notes_helpers as nh
from helpers import model_preprocessing as mp
from helpers import generative_models as gm




X, y = dl.get_midi_file_list('Rap/Gangster Rap')

print(X)

song_stream = dl.get_chordified_stream(X)
song_notes, notes_durations = nh.extract_notes_and_durations(song_stream, with_Rest=True)
note_map, reverse_note_map, length_notes = nh.create_feature_map(feature=song_notes)

duration_map, reverse_duration_map, length_duration = nh.create_feature_map(feature=notes_durations)
print(max(reverse_duration_map.keys()))
notes_features, notes_targets, duration_features, duration_targets = mp.prepare_note_duration_sequences(notes=song_notes, note_map=note_map, 
                                                                                                    durations=notes_durations, duration_map=duration_map,
                                                                                                    sequence_length=40, step=1)
model_inputs, model_outputs, test_inputs, test_outputs = mp.prepare_notes_duration_for_model(notes_features, notes_targets, duration_features, duration_targets,
                                                                    length_notes, length_duration, test_size=60)
model = gm.GenerativeEncodedModel(model_inputs=model_inputs, test_inputs=test_inputs,
                                         model_outputs=model_outputs, test_outputs=test_outputs)
model.create_network(length_notes, length_duration)
model.compile_model()
model.print_summary()
history = model.train_model(model_name='encoded_model', batch_size=100, epochs=10)
music, music_durations = model.generate_new(reverse_note_map, reverse_duration_map,
                                                 length_notes, length_duration, sample_size=120) 
melody = nh.create_melody_from_notes_durations(music, music_durations)
melody_stream = nh.create_midi_stream_from_notes(melody=melody)
dl.save_midi2file(melody_stream=melody_stream, file_name='encoded_test')
dl.convert_midi2audio(midi_file_name='encoded_test', output_filename='encoded_test_audio')