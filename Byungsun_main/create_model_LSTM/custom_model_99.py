import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.layers import Bidirectional, LSTM, Dense
from keras.layers import ReLU, Flatten


def CNN_99():
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(filters=128, kernel_size=2, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(filters=128, kernel_size=2, padding='same'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.build(input_shape=(None, 15, 99))
    return model


def LS_99():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(15, 99), dropout=0.3))
    model.add(LSTM(64, return_sequences=True, dropout=0.3))
    model.add(LSTM(64, return_sequences=True, dropout=0.3))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.build(input_shape=(None, 15, 99))
    return model


def BiLS_99():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(15, 99), dropout=0.3)))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(64, return_sequences=True, dropout=0.3)))
    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))

    model.build(input_shape=(None, 15, 99))
    return model