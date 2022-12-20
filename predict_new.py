from music21 import converter, stream
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

altrock_num_classes = 221

electro_num_classes = 14

glamrock_num_classes = 9122

soul_num_classes = 5606

def create_model(num_classes: int):
    model = Sequential([LSTM(256, input_shape=(40, 1),
                                    return_sequences=True),
                        Dropout(0.2),
                        LSTM(256),
                        Dense(256),
                        Dropout(0.2),
                        Dense(num_classes, activation='softmax')
                        ])

    model.compile(Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


test_model = create_model(electro_num_classes)

test_model.load_weights(filepath='./model/electro_model.h5')


 def generate_new(model, reverse_map: dict, normalization_factor: int, sample_size: int=50) -> list:
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


