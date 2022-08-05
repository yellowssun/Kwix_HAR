import tensorflow as tf
from keras.layers import Bidirectional, LSTM, Dense
import numpy as np


# class BiLAT_model(tf.keras.Model):
#     def __init__(self, units):
#         super(BiLAT_model, self).__init__()
#         self.input = tf.keras.Input(shape=())
#         self.L1 = Bidirectional(layer=LSTM(64, return_sequences=True, input_shape=x_data.shape[1:3])))
#         self.L2 = LSTM
#         self.L3 = LSTM
#         self.L4 = Dense
#         self.L5 = Dense(units)
#         self.L6 = Dense

    
#     def call(self, features, hidden):
#         hidden_with_time_axis = tf.expand_dims(hidden, 1)
#         score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
#         attention_weights = tf.nn.spftmax(self.V(score), axis=1)
#         context_vector = attention_weights * features
#         context_vector = tf.reduce_sum(context_vector, axis=1)

#         return context_vector, attention_weights


x = np.random.rand(10, 15, 47)
print(x.shape)
print(x[1:3].shape)