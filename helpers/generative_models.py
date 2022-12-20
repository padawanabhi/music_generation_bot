import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Dense, LSTM, Concatenate, Embedding, Input
import keras.backend as K
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, History

class GenerativeModel():
    def __init__(self, model_inputs: np.ndarray, test_inputs: np.ndarray, 
                    model_outputs: np.ndarray, test_outputs: np.ndarray) -> None:
        self.X_train = model_inputs.copy()
        self.X_test = test_inputs.copy()
        self.y_train = model_outputs.copy()
        self.y_test = test_outputs.copy()
        self.prediction = ""
        self.model_history = None
        self.model = None
        
        
    def create_network(self):

        self.model = Sequential([
                                LSTM(256, input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                                            return_sequences=True),
                                Dropout(0.2),
                                LSTM(256),
                                Dense(256),
                                Dropout(0.2),
                                Dense(self.y_train.shape[1], activation='softmax')
                                ])
        return self.model

    def compile_model(self):

        self.model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


    def print_summary(self):
        return self.model.summary()


    def train_model(self, model_name: str='default',batch_size: int=50, epochs: int=10, patience: int=10) -> History:

        K.clear_session()

        model_checkpoint = ModelCheckpoint(filepath=f'./model/{model_name}.h5', monitor='val_accuracy',
                                            save_weights_only=True  ,mode='max', save_best_only=True, verbose=1)
        
        model_stopping = EarlyStopping(monitor='val_accuracy', restore_best_weights=True, 
                            patience= patience, verbose=1)

        callback_list = [model_checkpoint, model_stopping]

        self.model_history = self.model.fit(self.X_train, self.y_train, batch_size=batch_size,
                                            epochs=epochs, validation_split=0.2, callbacks=callback_list)
        
        return self.model_history


    def generate_new(self, reverse_map: dict, normalization_factor: int, sample_size: int=50) -> list:
        seed = self.X_test[np.random.randint(0, len(self.X_test)-1)]
        predicted_notes = []
        for i in range(sample_size):
            seed = seed.reshape(1, self.X_test.shape[1], 1)
            prediction = self.model.predict(seed, verbose=1)
            prediction = np.log(prediction)
            exp_pred = np.exp(prediction)
            prediction = exp_pred/np.sum(exp_pred)
            pred_index = np.argmax(prediction)
            pred_index_n = pred_index/normalization_factor
            predicted_notes.append(pred_index)
            self.prediction = [reverse_map[char] for char in predicted_notes]
            seed = np.insert(seed[0], len(seed[0]), pred_index_n)
            seed = seed[1:]

        print(self.prediction)
        return self.prediction





class GenerativeEmbeddedModel(GenerativeModel):
    def __init__(self, model_inputs: np.ndarray, test_inputs: np.ndarray, model_outputs: np.ndarray, test_outputs: np.ndarray) -> None:
        super().__init__(model_inputs, test_inputs, model_outputs, test_outputs)
        self.model = None
        self.notes_prediction = []
        self.durations_prediction = []
        self.model_history = None
    
    def create_network(self, length_notes: int, length_duration: int):
        """_summary_

        Args:
            length_notes (int): _description_
            length_duration (int): _description_

        Returns:
            _type_: _description_
        """
        duration_in = Input(shape=(None,))
        notes_in = Input(shape=(None,))
        
        embedded_notes = Embedding(length_notes, 100)(notes_in)
        embedded_durations = Embedding(length_duration, 100)(duration_in)

        inputs = Concatenate()([embedded_notes, embedded_durations])
        
        lstm_output = LSTM(512, return_sequences=True)(inputs)
        lstm2_output = LSTM(256)(lstm_output)
        notes_out = Dense(length_notes, activation='softmax', name='notes')(lstm2_output)
        duration_out = Dense(length_duration, activation='softmax', name='duration')(lstm2_output)

        self.model = Model([notes_in, duration_in], [notes_out, duration_out])

        return self.model


    def compile_model(self):

        self.model.compile(optimizer=RMSprop(0.001), loss=['categorical_crossentropy', 'categorical_crossentropy'],
                    metrics=['accuracy'])

    def train_model(self, model_name: str = 'default', batch_size: int = 50, epochs: int = 50, patience: int=10) -> History:
        return super().train_model(model_name, batch_size, epochs, patience)


    def generate_new(self, reverse_note_map: dict, reverse_duration_map: dict,
                            notes_factor: int, duration_factor: int, sample_size: int=40):
        """_summary_

        Args:
            model (_type_): _description_
            x_test (list): _description_
            reverse_note_map (dict): _description_
            reverse_duration_map (dict): _description_
            notes_factor (int): _description_
            sample_size (int, optional): _description_. Defaults to 50.

        Returns:
            _type_: _description_
        """
        seed_index = np.random.randint(0, len(self.X_test)-1)
        seed = [self.X_test[0][seed_index], self.X_test[1][seed_index]]
        predicted_notes = []
        predicted_durations = []
        for i in range(sample_size):
            test_note = np.reshape(seed[0], (seed[0].shape[0], 1))
            test_duration = np.reshape(seed[1], (seed[0].shape[0], 1))
            seed = [test_note, test_duration]
            notes_prediction, duration_prediction = self.model.predict(seed, verbose=1)
            notes_prediction = np.log(notes_prediction)
            duration_prediction = np.log(duration_prediction)
            exp_note_pred = np.exp(notes_prediction)
            exp_duration_pred = np.exp(duration_prediction)
            notes_prediction = exp_note_pred/np.sum(exp_note_pred)
            duration_prediction = exp_duration_pred/np.sum(exp_duration_pred)
            pred_note_index = np.argmax(notes_prediction)
            pred_index_notes = pred_note_index/notes_factor
            pred_duration_index = np.argmax(duration_prediction)
            pred_index_duration = pred_duration_index/duration_factor
            predicted_notes.append(pred_note_index)
            predicted_durations.append(pred_duration_index)
            print(predicted_notes, predicted_durations)
            print(pred_index_notes, pred_index_duration)
            self.notes_prediction = [reverse_note_map[char] for char in predicted_notes]
            self.durations_prediction = [reverse_duration_map[dur] for dur in predicted_durations]
            note_seed = np.insert(seed[0], seed[0].shape[0], pred_index_notes)
            duration_seed = np.insert(seed[1], seed[1].shape[0], pred_index_duration)
            seed = [note_seed[1:], duration_seed[1:]]

        print(self.notes_prediction)
        print(self.durations_prediction)
        return self.notes_prediction, self.durations_prediction