import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, Conv2D, BatchNormalization, GlobalAveragePooling1D, GlobalAveragePooling2D
from keras.layers import Bidirectional, LSTM, Dense, SimpleRNN
from keras.layers import ReLU, Dropout, Input, Concatenate, Flatten

def two_input_BiLS():
    input_landmark = Input(shape=(15, 51), name='landmark_input')
    input_angle = Input(shape=(15, 10), name='angle_input')

    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, name='bidirectional_L1'))(input_landmark)
    x = Bidirectional(LSTM(128, return_sequences=True, name='bidirectional_L2'))(x)
    x = Flatten(x)
    # x = Model(inputs=input_landmark, outputs=x)

    y = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, name='bidirectional_A1'))(input_angle)
    y = Bidirectional(LSTM(128, return_sequences=True, name='bidirectional_A2'))(y)
    y = Flatten(y)
    # y = Model(inputs=input_landmark, outputs=y)    

    concatenate = Concatenate()([x, y])

    z = Dense(32, activation='relu')(concatenate)
    z = Dense(16, activation='relu')(z)
    z = Dense(2, activation='softmax')(z)

    model = Model(inputs=[input_landmark, input_angle], outputs=z)
    return model


# def two_input_CNN():
#     input_landmark = Input(shape=(15, 51), name='landmark_input')
#     input_angle = Input(shape=(15, 10), name='angle_input')

#     x = Conv2D(filters=64, kernel_size=(3, 2), stride=(3, 1), padding='valid')(input_landmark)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = Conv1D(filters=64, kernel_size=4, padding='valid')(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)
#     x = Conv1D(filters=64, kernel_size=4, padding='valid')(x)
#     x = BatchNormalization()(x)
#     x = ReLU()(x)

#     y = Conv1D(filters=64, kernel_size=4, padding='same')(y)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     y = Conv1D(filters=64, kernel_size=4, padding='same')(y)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)
#     y = Conv1D(filters=64, kernel_size=4, padding='same')(y)
#     y = BatchNormalization()(y)
#     y = ReLU()(y)

#     concatenate = Concatenate()([x, y])
#     pooling = GlobalAveragePooling1D()(concatenate)

#     z = Dense(32, activation='relu')(pooling)
#     z = Dense(16, activation='relu')(z)
#     z = Dense(2, activation='softmax')(z)

#     model = Model(inputs=[input_landmark, input_angle], outputs=z)
#     return model


def CNN():
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(filters=64, kernel_size=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(filters=32, kernel_size=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(filters=16, kernel_size=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(GlobalAveragePooling1D())
    model.add(Dense(2, activation='softmax'))

    model.build(input_shape=(None, 15, 99))
    # plot_model(model, to_file='CNN_model.png', show_shapes=True)

    return model


def RNN():
    model = Sequential()
    model.add(SimpleRNN(32, return_sequences=True, input_shape=(15, 99), dropout=0.3))
    model.add(SimpleRNN(64, return_sequences=True, dropout=0.3))
    model.add(SimpleRNN(64, return_sequences=True, dropout=0.3))
    model.add(SimpleRNN(32))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.build(input_shape=(None, 15, 99))
    # plot_model(model, to_file='BiLS_model.png', show_shapes=True)
    return model

def BiLS():
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(15, 99), dropout=0.3)))
    model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.3)))
    model.add(Bidirectional(LSTM(32)))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.build(input_shape=(None, 15, 99))
    # plot_model(model, to_file='BiLS_model.png', show_shapes=True)
    return model


def LS():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(15, 99), dropout=0.3))
    model.add(LSTM(64, return_sequences=True, dropout=0.3))
    model.add(LSTM(32))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.build(input_shape=(None, 15, 99))
    # plot_model(model, to_file='BiLS_model.png', show_shapes=True)
    return model


def LS_CNN():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(15, 99), dropout=0.3))
    model.add(LSTM(64, return_sequences=True, dropout=0.3))
    
    model.add(Conv1D(filters=64, kernel_size=1, padding='valid'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(filters=32, kernel_size=1, padding='valid'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(GlobalAveragePooling1D())
    model.add(Dense(2, activation='softmax'))
    
    model.build(input_shape=(15, 99))
    return model


def CNN_LS():
    model = Sequential()
    model.add(Conv1D(filters=124, kernel_size=1, padding='valid'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Conv1D(filters=64, kernel_size=1, padding='valid'))
    model.add(Dropout(rate=0.5))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(LSTM(64, return_sequences=True, dropout=0.3))
    model.add(LSTM(32, dropout=0.3))

    model.add(Dense(2, activation='softmax'))
    model.build(input_shape=(None, 15, 99))
    return model

    # model.add(Conv1D(filters=32, kernel_size=4, padding='valid'))
    # model.add(Dropout(rate=0.5))
    # model.add(BatchNormalization())
    # model.add(ReLU())

    # model.add(LSTM(32, return_sequences=True, dropout=0.3))

    # model.add(Conv1D(filters=16, kernel_size=4, padding='valid'))
    # model.add(Dropout(rate=0.5))
    # model.add(BatchNormalization())
    # model.add(ReLU())

    # model.add(LSTM(16, return_sequences=True, dropout=0.3))

    # model.add(Conv1D(filters=8, kernel_size=4, padding='valid'))
    # model.add(Dropout(rate=0.5))
    # model.add(BatchNormalization())
    # model.add(ReLU())

    # model.add(Bidirectional(LSTM(8, return_sequences=True, dropout=0.3)))

    # model.add(Dense(2, activation='softmax'))
    
    # model.build(input_shape=(None, 15, 99))
    # return model